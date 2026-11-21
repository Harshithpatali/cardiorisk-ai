"""Leakage-aware selective verification cascade (research only).

Stage 1: existing PyTorch MLP screens at a development-OOF recall threshold.
Scores in [gate, 0.50) are routed to Stage 2; scores >= 0.50 bypass Stage 2
as positive. Scores below gate are negative. Stage 2 compares calibrated
XGBoost and LightGBM models trained only on OOF-routed development rows.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from config import ARTIFACT_DIR, DATA_PATH, RANDOM_SEED, TEST_SIZE
from data_split import split_holdout
from evaluation import evaluate_predictions
from model_training import predict_probabilities, train_fixed_epochs
from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame
from bayesian_mlp_cascade import RECALL_FLOORS, paired_bootstrap_differences, select_recall_threshold

FOLDS = 5
GATE_RECALL = 0.995
ROUTE_CEILING = 0.50
BOOTSTRAPS = 2000


def make_estimator(name, seed):
    if name == "xgboost":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
            n_estimators=350, max_depth=3, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=5, reg_lambda=5.0,
            scale_pos_weight=1.0, tree_method="hist", n_jobs=2, random_state=seed)
    if name == "lightgbm":
        return lgb.LGBMClassifier(objective="binary", n_estimators=350,
            max_depth=5, num_leaves=15, learning_rate=0.03, subsample=0.85,
            subsample_freq=1, colsample_bytree=0.85, reg_lambda=5.0,
            min_child_samples=30, class_weight=None, verbosity=-1,
            n_jobs=2, random_state=seed)
    raise ValueError(f"Unknown second-stage model: {name}")


def calibrated_estimator(name, seed):
    # Sigmoid calibration is fitted only inside the supplied development fold.
    return CalibratedClassifierCV(estimator=make_estimator(name, seed), method="sigmoid", cv=3)


def stage2_oof_predictions(frame, target, model_name, seed):
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    oof = np.full(len(target), np.nan, dtype=float)
    rows = []
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(frame, target), 1):
        fit_frame = frame.iloc[fit_idx].reset_index(drop=True)
        hold_frame = frame.iloc[hold_idx].reset_index(drop=True)
        prep = fit_preprocessor(fit_frame)
        x_fit, x_hold = transform_frame(fit_frame, prep), transform_frame(hold_frame, prep)
        model = calibrated_estimator(model_name, seed + fold)
        model.fit(x_fit, target[fit_idx])
        oof[hold_idx] = model.predict_proba(x_hold)[:, 1]
        rows.append({"model": model_name, "fold": fold,
            "training_rows": len(fit_idx), "validation_rows": len(hold_idx)})
    if np.isnan(oof).any():
        raise RuntimeError(f"{model_name} OOF predictions are incomplete")
    return oof, rows


def choose_cascade_threshold(y, stage1_oof, stage2_oof, gate, recall_floor):
    direct = stage1_oof >= ROUTE_CEILING
    routed = (stage1_oof >= gate) & ~direct
    direct_tp = int(np.sum(direct & (y == 1)))
    direct_fp = int(np.sum(direct & (y == 0)))
    total_pos = int(np.sum(y == 1))
    candidates = np.unique(np.r_[0.0, stage2_oof, 1.0])
    feasible = []
    for threshold in candidates:
        s2_positive = routed & (stage2_oof >= threshold)
        tp = direct_tp + int(np.sum(s2_positive & (y == 1)))
        fp = direct_fp + int(np.sum(s2_positive & (y == 0)))
        recall = tp / max(total_pos, 1)
        if recall + 1e-12 < recall_floor:
            continue
        # Beta(1,1) posterior lower 95% bound; exact beta quantile.
        from scipy.stats import beta
        lower = float(beta.ppf(0.05, tp + 1, fp + 1))
        feasible.append((lower, tp / max(tp + fp, 1), threshold, recall, tp, fp))
    if not feasible:
        raise RuntimeError(f"No Stage-2 threshold achieves end-to-end recall floor {recall_floor:.3f}")
    best = max(feasible, key=lambda z: (z[0], z[1], -z[2]))
    return {"threshold": float(best[2]), "cascade_recall_oof": float(best[3]),
        "precision_posterior_mean": float((best[4]+1)/(best[4]+best[5]+2)),
        "precision_lower_credible_bound": float(best[0]),
        "cascade_precision_oof": float(best[1]),
        "direct_positive_rows": int(np.sum(direct)), "routed_rows": int(np.sum(routed))}


def run():
    artifacts = Path(ARTIFACT_DIR)
    baseline_path, oof_path = artifacts / "metrics.json", artifacts / "mlp_oof_predictions.csv"
    if not baseline_path.exists() or not oof_path.exists():
        raise FileNotFoundError("Run train.py first to create MLP artifacts.")
    baseline = json.loads(baseline_path.read_text())
    baseline_oof = pd.read_csv(oof_path)
    raw = pd.read_csv(DATA_PATH, sep=";")
    raw_bmi = raw["weight"] / ((raw["height"] / 100.0) ** 2)
    raw_pulse = raw["ap_hi"] - raw["ap_lo"]
    raw_keep = (
        raw["cardio"].notna()
        & raw["cardio"].isin([0, 1])
        & raw["ap_hi"].between(70, 250)
        & raw["ap_lo"].between(40, 180)
        & (raw["ap_hi"] > raw["ap_lo"])
        & raw["height"].between(120, 220)
        & raw["weight"].between(35, 200)
        & raw_bmi.between(12, 60)
    )
    raw_kept = raw.loc[raw_keep, ["id", "age"]].reset_index(drop=True)
    data = clean_dataset(DATA_PATH)
    if len(raw_kept) != len(data):
        raise RuntimeError("Raw-to-cleaned row alignment failed; refusing to write a misleading Stage-1 ledger.")
    x, y = data[FEATURE_COLUMNS].reset_index(drop=True), data["cardio"].astype(int).to_numpy()
    ledger_x = pd.concat([raw_kept, x], axis=1)
    x_dev, x_test, y_dev, y_test = split_holdout(x, y, TEST_SIZE, RANDOM_SEED)
    ledger_dev, ledger_test, ledger_y_dev, ledger_y_test = split_holdout(ledger_x, y, TEST_SIZE, RANDOM_SEED)
    if not np.array_equal(ledger_y_dev, y_dev) or not np.array_equal(ledger_y_test, y_test):
        raise RuntimeError("Ledger split does not match model split.")
    if len(baseline_oof) != len(y_dev) or not np.array_equal(baseline_oof["target"].to_numpy(dtype=int), y_dev):
        raise RuntimeError("MLP OOF predictions do not match deterministic development split.")

    p1_dev = baseline_oof["oof_probability"].to_numpy(float)
    gate, gate_recall = select_recall_threshold(y_dev, p1_dev, GATE_RECALL)
    prep1 = fit_preprocessor(x_dev)
    dev_matrix, test_matrix = transform_frame(x_dev, prep1), transform_frame(x_test, prep1)
    stage1 = train_fixed_epochs(dev_matrix, y_dev,
        int(baseline["cross_validation"]["final_refit_epochs"]), RANDOM_SEED, 512)
    p1_test = predict_probabilities(stage1, test_matrix)

    dev_route = (p1_dev >= gate) & (p1_dev < ROUTE_CEILING)
    test_route = (p1_test >= gate) & (p1_test < ROUTE_CEILING)
    dev_direct, test_direct = p1_dev >= ROUTE_CEILING, p1_test >= ROUTE_CEILING
    routed_x = x_dev.loc[dev_route].reset_index(drop=True)
    routed_y = y_dev[dev_route]
    if len(routed_y) < 30 or len(np.unique(routed_y)) != 2 or min(np.bincount(routed_y)) < FOLDS * 3:
        raise RuntimeError(f"Insufficient routed Stage-1 borderline rows for nested calibration: n={len(routed_y)}, class_counts={np.bincount(routed_y).tolist()}")

    # Persist every original feature and its actual/OOF-or-holdout prediction.
    # Only development OOF borderline rows are eligible for Stage-2 fitting.
    dev_snapshot = ledger_dev.copy()
    dev_snapshot["actual"] = y_dev
    dev_snapshot["stage1_prediction"] = (p1_dev >= gate).astype(int)
    dev_snapshot["stage1_positive_confidence"] = p1_dev
    dev_snapshot["stage1_split"] = "development_oof"
    dev_snapshot["stage1_route"] = np.where(dev_direct, "direct_positive",
        np.where(dev_route, "stage2_borderline", "stage1_negative"))
    test_snapshot = ledger_test.copy()
    test_snapshot["actual"] = y_test
    test_snapshot["stage1_prediction"] = (p1_test >= gate).astype(int)
    test_snapshot["stage1_positive_confidence"] = p1_test
    test_snapshot["stage1_split"] = "untouched_test_final_model"
    test_snapshot["stage1_route"] = np.where(test_direct, "direct_positive",
        np.where(test_route, "stage2_borderline", "stage1_negative"))
    snapshot = pd.concat([dev_snapshot, test_snapshot], ignore_index=True)
    snapshot["stage2_training_eligible"] = snapshot["stage1_split"].eq("development_oof") & snapshot["stage1_route"].eq("stage2_borderline")
    snapshot.to_csv(artifacts / "stage1_prediction_dataset.csv", index=False)

    models, all_folds, flat_rows = [], [], []
    for mi, name in enumerate(("xgboost", "lightgbm")):
        seed = RANDOM_SEED + 20000 * (mi + 1)
        s2_oof, fold_rows = stage2_oof_predictions(routed_x, routed_y, name, seed)
        all_folds.extend(fold_rows)
        result = {"model": name, "status": "research_only_not_promoted",
            "stage1_gate_threshold": float(gate), "stage1_route_ceiling": ROUTE_CEILING,
            "stage1_gate_oof_recall": float(gate_recall),
            "stage2_development_rows": int(len(routed_y)),
            "stage2_development_positive_rate": float(np.mean(routed_y)),
            "recall_target_results": []}
        prep2 = fit_preprocessor(routed_x)
        fit_x = transform_frame(routed_x, prep2)
        final = calibrated_estimator(name, seed + 999)
        final.fit(fit_x, routed_y)
        test_x = transform_frame(x_test.loc[test_route].reset_index(drop=True), prep2)
        s2_test = final.predict_proba(test_x)[:, 1] if len(test_x) else np.array([])
        stage2_oof_full = np.zeros(len(y_dev), dtype=float)
        stage2_oof_full[dev_route] = s2_oof
        for ti, floor in enumerate(RECALL_FLOORS):
            choice = choose_cascade_threshold(y_dev, p1_dev,
                stage2_oof_full, gate, floor)
            # Build the end-to-end Stage-2 score vector. Direct positives bypass Stage 2.
            final_scores = np.zeros(len(y_test), dtype=float)
            final_scores[test_direct] = 1.0
            final_scores[np.flatnonzero(test_route)] = s2_test
            cascade_pred = (test_direct | (test_route & (s2_test >= choice["threshold"]))).astype(int)
            cascade_metrics = evaluate_predictions(y_test, cascade_pred.astype(float), 0.5)
            baseline_threshold, baseline_recall = select_recall_threshold(y_dev, p1_dev, choice["cascade_recall_oof"])
            baseline_metrics = evaluate_predictions(y_test, p1_test, baseline_threshold)
            baseline_pred = (p1_test >= baseline_threshold).astype(int)
            paired = paired_bootstrap_differences(y_test, baseline_pred, cascade_pred,
                n_bootstrap=BOOTSTRAPS, seed=seed + 3000 + ti)
            row = {"model": name, "recall_target": float(floor),
                "cascade_oof_recall": choice["cascade_recall_oof"],
                "baseline_oof_recall": float(baseline_recall),
                "stage2_threshold": choice["threshold"],
                "bayesian_precision_oof_mean": choice["precision_posterior_mean"],
                "bayesian_precision_oof_lower_95": choice["precision_lower_credible_bound"],
                "stage2_routed_test_rows": int(np.sum(test_route)),
                "stage2_test_brier_score": float(brier_score_loss(y_test[test_route], s2_test)) if np.sum(test_route) else None,
                "baseline_test_precision": baseline_metrics["precision"],
                "cascade_test_precision": cascade_metrics["precision"],
                "baseline_test_recall": baseline_metrics["recall"],
                "cascade_test_recall": cascade_metrics["recall"],
                "baseline_test_specificity": baseline_metrics["specificity"],
                "cascade_test_specificity": cascade_metrics["specificity"],
                "baseline_test_f1": baseline_metrics["f1"],
                "cascade_test_f1": cascade_metrics["f1"],
                "baseline_test_false_positives": int(baseline_metrics["confusion_matrix"][0][1]),
                "cascade_test_false_positives": int(cascade_metrics["confusion_matrix"][0][1]),
                "baseline_test_false_negatives": int(baseline_metrics["confusion_matrix"][1][0]),
                "cascade_test_false_negatives": int(cascade_metrics["confusion_matrix"][1][0]),
                "paired_bootstrap_differences": paired}
            result["recall_target_results"].append(row)
            flat_rows.append({k:v for k,v in row.items() if k != "paired_bootstrap_differences"})
        models.append(result)

    report = {"experiment": "High-recall MLP with selective borderline XGBoost/LightGBM verification",
        "status": "research_only_not_promoted_to_deployed_inference",
        "test_rows": int(len(y_test)), "stage1_gate_threshold": float(gate),
        "stage1_route_ceiling": ROUTE_CEILING, "stage1_gate_oof_recall": float(gate_recall),
        "stage1": "Existing PyTorch MLP unchanged. OOF development predictions determine high-recall gate.",
        "routing": "Below gate: negative; gate <= confidence < 0.50: Stage 2; confidence >= 0.50: direct positive bypass.",
        "stage2": "XGBoost and LightGBM trained only on Stage-1 OOF borderline development rows; each outer fold uses sigmoid calibration fitted inside that fold.",
        "dataset_artifact": "stage1_prediction_dataset.csv contains original features, actual labels, Stage-1 prediction/confidence, split and route. Only development OOF borderline rows are Stage-2 training eligible.",
        "threshold_selection": "Beta(1,1) posterior precision lower 95% bound subject to end-to-end OOF recall floors of 95%, 97%, and 98%.",
        "comparison": "Single-stage MLP matched to cascade OOF recall; both evaluated on identical untouched test rows.",
        "bootstrap": "2,000 paired test-row bootstrap resamples; conditional on fitted models and selected thresholds.",
        "models": models,
        "warning": "Educational portfolio research only; cross-sectional labels are not future risk and this is not clinically validated or for diagnosis, screening, treatment or care decisions."}
    (artifacts / "tree_stage2_cascade_comparison.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(flat_rows).to_csv(artifacts / "tree_stage2_cascade_comparison.csv", index=False)
    pd.DataFrame(all_folds).to_csv(artifacts / "tree_stage2_cascade_folds.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
