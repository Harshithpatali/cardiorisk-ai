"""Development-only CatBoost/XGBoost comparison and sigmoid-calibration audit.

Five-fold outer OOF evaluation with fold-fitted preprocessing. Sigmoid
calibration is fitted only inside each outer training fold. No holdout data is
read and no candidate is promoted to deployed inference.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACT_DIR, DATA_PATH, RANDOM_SEED, TEST_SIZE
from data_split import split_holdout
from operating_point_analysis import operating_point_for_recall_floor
from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame

FOLDS = 5
RECALL_FLOORS = (0.95, 0.90, 0.85)
CALIBRATION_FOLDS = 3
ECE_BINS = 10


def expected_calibration_error(y, p, bins=ECE_BINS):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    rows = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.any():
            observed, confidence = float(y[mask].mean()), float(p[mask].mean())
            total += float(mask.mean()) * abs(observed - confidence)
            rows.append({"bin_lower": float(edges[i]), "bin_upper": float(edges[i + 1]),
                         "count": int(mask.sum()), "mean_score": confidence,
                         "observed_rate": observed})
    return float(total), rows


def make_estimators():
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    return {
        "catboost": CatBoostClassifier(
            iterations=450, depth=5, learning_rate=0.04, loss_function="Logloss",
            l2_leaf_reg=5.0, random_seed=RANDOM_SEED, verbose=False,
            allow_writing_files=False, thread_count=2,
        ),
        "xgboost": XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=2, random_state=RANDOM_SEED,
        ),
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def evaluate(y, p, fold_ids, model_name, probability_type):
    ece, bins = expected_calibration_error(y, p)
    floor_rows = []
    for floor in RECALL_FLOORS:
        point = operating_point_for_recall_floor(y, p, floor)
        fold_specificity, fold_recall = [], []
        for fold in sorted(np.unique(fold_ids)):
            mask = fold_ids == fold
            pred = (p[mask] >= point["threshold"]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[mask], pred, labels=[0, 1]).ravel()
            fold_specificity.append(float(tn / max(tn + fp, 1)))
            fold_recall.append(float(recall_score(y[mask], pred, zero_division=0)))
        floor_rows.append({
            "model": model_name, "probability_type": probability_type, **point,
            "specificity_fold_ci95": mean_ci(fold_specificity),
            "recall_fold_ci95": mean_ci(fold_recall),
        })
    return {
        "model": model_name, "probability_type": probability_type,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_bins": ece,
        "reliability_bins": bins,
        "operating_points": floor_rows,
    }


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    data = clean_dataset(DATA_PATH).reset_index(drop=True)
    x, _, y, _ = split_holdout(data[FEATURE_COLUMNS], data["cardio"].astype(int).to_numpy(),
                               TEST_SIZE, RANDOM_SEED)
    x = x.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_ids = np.zeros(len(y), dtype=int)
    score_store = {}
    fold_rows = []

    mlp_path = ARTIFACT_DIR / "specificity_oof_predictions.csv"
    if not mlp_path.exists():
        raise FileNotFoundError("Run specificity_investigation.py before this experiment")
    mlp = pd.read_csv(mlp_path)
    if len(mlp) != len(y) or not np.array_equal(mlp["target"].to_numpy(), y):
        raise RuntimeError("MLP OOF rows do not align with this development split")
    if not np.array_equal(mlp["fold"].to_numpy(), np.concatenate([
        np.full(len(hold), fold, dtype=int)
        for fold, (_, hold) in enumerate(splitter.split(x, y), start=1)
    ])):
        raise RuntimeError("MLP OOF fold assignments do not match this experiment")
    score_store["mlp_reference"] = {
        "uncalibrated": mlp["mlp_positive_weight_1.00__probability"].to_numpy(dtype=float)
    }

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x, y), start=1):
        fold_ids[hold_idx] = fold
        x_fit, x_hold = x.iloc[fit_idx].reset_index(drop=True), x.iloc[hold_idx].reset_index(drop=True)
        y_fit = y[fit_idx]
        prep = fit_preprocessor(x_fit)
        fit_matrix, hold_matrix = transform_frame(x_fit, prep), transform_frame(x_hold, prep)
        for name, estimator in make_estimators().items():
            estimator.fit(fit_matrix, y_fit)
            raw = estimator.predict_proba(hold_matrix)[:, 1]
            calibrated = CalibratedClassifierCV(
                estimator=make_estimators()[name], method="sigmoid",
                cv=CALIBRATION_FOLDS,
            )
            calibrated.fit(fit_matrix, y_fit)
            calibrated_scores = calibrated.predict_proba(hold_matrix)[:, 1]
            score_store.setdefault(name, {
                "uncalibrated": np.full(len(y), np.nan),
                "sigmoid_calibrated": np.full(len(y), np.nan),
            })
            score_store[name]["uncalibrated"][hold_idx] = raw
            score_store[name]["sigmoid_calibrated"][hold_idx] = calibrated_scores
            fold_rows.append({
                "fold": fold, "model": name,
                "uncalibrated_roc_auc": float(roc_auc_score(y[hold_idx], raw)),
                "calibrated_roc_auc": float(roc_auc_score(y[hold_idx], calibrated_scores)),
                "uncalibrated_brier": float(brier_score_loss(y[hold_idx], raw)),
                "calibrated_brier": float(brier_score_loss(y[hold_idx], calibrated_scores)),
            })
        print(f"Completed CatBoost/XGBoost fold {fold}/{FOLDS}")

    oof = pd.DataFrame({"target": y, "fold": fold_ids})
    summaries, operating_rows = [], []
    for name, variants in score_store.items():
        for probability_type, scores in variants.items():
            if np.isnan(scores).any():
                raise RuntimeError(f"Incomplete OOF predictions: {name}/{probability_type}")
            oof[f"{name}__{probability_type}"] = scores
            result = evaluate(y, scores, fold_ids, name, probability_type)
            operating_rows.extend(result.pop("operating_points"))
            summaries.append(result)

    oof.to_csv(ARTIFACT_DIR / "boosting_calibration_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ARTIFACT_DIR / "boosting_calibration_folds.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(ARTIFACT_DIR / "boosting_calibration_operating_points.csv", index=False)
    payload = {
        "method": "5-fold stratified development OOF; fold-fitted preprocessing; sigmoid calibration fitted with 3-fold CV inside each outer training fold",
        "scope": "development partition only; final holdout not loaded, evaluated, or used for model/threshold selection",
        "models": summaries,
        "operating_points": operating_rows,
        "fold_results": fold_rows,
        "selection_policy": "No automatic model or threshold promotion. Compare ranking, calibration and specificity at matched recall; independent confirmation is required.",
        "limitations": [
            "Internal cross-validation on one cross-sectional dataset only.",
            "OOF threshold selection and reporting on the same pooled OOF scores are exploratory and optimistic; use nested threshold selection or an independent validation cohort for confirmation.",
            "Calibration is an internal probability-reliability diagnostic, not clinical calibration.",
            "This dataset does not measure future cardiovascular events; no clinical validity or utility is established.",
        ],
    }
    (ARTIFACT_DIR / "boosting_calibration_study.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "models": [{"model": r["model"], "probability_type": r["probability_type"],
                    "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                    "brier_score": r["brier_score"], "ece_10_bins": r["ece_10_bins"]}
                   for r in summaries],
        "operating_point_rows": len(operating_rows),
        "artifacts": ["boosting_calibration_study.json", "boosting_calibration_oof_predictions.csv",
                      "boosting_calibration_folds.csv", "boosting_calibration_operating_points.csv"],
    }, indent=2))


if __name__ == "__main__":
    main()
