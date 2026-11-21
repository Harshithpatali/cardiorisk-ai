"""Research-only XGBoost and LightGBM second-stage cascade comparison.

The deployed Stage-1 PyTorch MLP is unchanged. Tree models are trained only
on Stage-1-positive development rows, with cross-fitted Stage-2 predictions.
Thresholds are selected on development OOF scores; the untouched test set is
used once for final matched-recall comparisons.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from config import ARTIFACT_DIR, DATA_PATH, RANDOM_SEED, TEST_SIZE
from data_split import split_holdout
from evaluation import evaluate_predictions
from model_training import predict_probabilities, train_fixed_epochs
from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame
from bayesian_mlp_cascade import (
    RECALL_FLOORS,
    bayesian_precision_threshold,
    paired_bootstrap_differences,
    select_recall_threshold,
)

FOLDS = 5
GATE_RECALL = 0.995
BOOTSTRAPS = 2000


def make_estimator(name, seed):
    if name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=350, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=5, reg_lambda=5.0,
            scale_pos_weight=1.0, tree_method="hist",
            n_jobs=2, random_state=seed,
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=350, max_depth=5,
            num_leaves=15, learning_rate=0.03,
            subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=5.0,
            min_child_samples=30, class_weight=None,
            verbosity=-1, n_jobs=2, random_state=seed,
        )
    raise ValueError(f"Unknown second-stage model: {name}")


def stage2_oof_predictions(frame, target, model_name, seed):
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    oof = np.full(len(target), np.nan, dtype=float)
    fold_rows = []
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(frame, target), start=1):
        fit_frame = frame.iloc[fit_idx].reset_index(drop=True)
        hold_frame = frame.iloc[hold_idx].reset_index(drop=True)
        prep = fit_preprocessor(fit_frame)
        x_fit = transform_frame(fit_frame, prep)
        x_hold = transform_frame(hold_frame, prep)
        model = make_estimator(model_name, seed + fold)
        model.fit(x_fit, target[fit_idx])
        oof[hold_idx] = model.predict_proba(x_hold)[:, 1]
        fold_rows.append({
            "fold": fold,
            "training_rows": int(len(fit_idx)),
            "validation_rows": int(len(hold_idx)),
            "model": model_name,
        })
    if np.isnan(oof).any():
        raise RuntimeError(f"{model_name} OOF predictions are incomplete")
    return oof, fold_rows


def run():
    artifacts = Path(ARTIFACT_DIR)
    baseline_path = artifacts / "metrics.json"
    baseline_oof_path = artifacts / "mlp_oof_predictions.csv"
    if not baseline_path.exists() or not baseline_oof_path.exists():
        raise FileNotFoundError("Run train.py first to create MLP artifacts.")

    baseline = json.loads(baseline_path.read_text())
    baseline_oof = pd.read_csv(baseline_oof_path)
    data = clean_dataset(DATA_PATH)
    x = data[FEATURE_COLUMNS].reset_index(drop=True)
    y = data["cardio"].astype(int).to_numpy()
    x_dev, x_test, y_dev, y_test = split_holdout(x, y, TEST_SIZE, RANDOM_SEED)
    if len(baseline_oof) != len(y_dev) or not np.array_equal(
        baseline_oof["target"].to_numpy(dtype=int), y_dev
    ):
        raise RuntimeError("MLP OOF predictions do not match the deterministic development split.")

    # Freeze the original Stage-1 MLP exactly as in the existing cascade study.
    stage1_oof = baseline_oof["oof_probability"].to_numpy(dtype=float)
    prep1 = fit_preprocessor(x_dev)
    dev_matrix = transform_frame(x_dev, prep1)
    test_matrix = transform_frame(x_test, prep1)
    stage1_model = train_fixed_epochs(
        dev_matrix, y_dev, int(baseline["cross_validation"]["final_refit_epochs"]),
        RANDOM_SEED, 512,
    )
    stage1_test = predict_probabilities(stage1_model, test_matrix)
    gate_threshold, gate_oof_recall = select_recall_threshold(
        y_dev, stage1_oof, GATE_RECALL
    )
    dev_gate = stage1_oof >= gate_threshold
    test_gate = stage1_test >= gate_threshold
    dev_idx = np.flatnonzero(dev_gate)
    test_idx = np.flatnonzero(test_gate)
    stage2_frame = x_dev.iloc[dev_idx].reset_index(drop=True)
    stage2_target = y_dev[dev_idx]
    stage1_dev_tp = int(np.sum(dev_gate & (y_dev == 1)))
    total_dev_positives = int(np.sum(y_dev == 1))

    model_results = []
    all_folds = []
    matched_rows = []
    for model_index, model_name in enumerate(("xgboost", "lightgbm")):
        seed = RANDOM_SEED + 20000 * (model_index + 1)
        stage2_oof, fold_rows = stage2_oof_predictions(
            stage2_frame, stage2_target, model_name, seed
        )
        all_folds.extend(fold_rows)
        model_result = {
            "model": model_name,
            "status": "research_only_not_promoted",
            "stage1_gate_threshold": float(gate_threshold),
            "stage1_gate_oof_recall": float(gate_oof_recall),
            "stage2_development_rows": int(len(dev_idx)),
            "stage2_development_positive_rate": float(np.mean(stage2_target)),
            "recall_target_results": [],
        }

        for target_index, recall_floor in enumerate(RECALL_FLOORS):
            choice = bayesian_precision_threshold(
                stage2_target, stage2_oof, stage1_dev_tp,
                total_dev_positives, recall_floor,
            )
            prep2 = fit_preprocessor(stage2_frame)
            fit_matrix = transform_frame(stage2_frame, prep2)
            final_model = make_estimator(
                model_name, seed + 1000 + target_index
            )
            final_model.fit(fit_matrix, stage2_target)

            cascade_scores = np.zeros(len(y_test), dtype=float)
            if len(test_idx):
                test_stage2_frame = x_test.iloc[test_idx].reset_index(drop=True)
                test_stage2_matrix = transform_frame(test_stage2_frame, prep2)
                cascade_scores[test_idx] = final_model.predict_proba(test_stage2_matrix)[:, 1]

            cascade_metrics = evaluate_predictions(
                y_test, cascade_scores, choice["threshold"]
            )
            baseline_threshold, baseline_oof_recall = select_recall_threshold(
                y_dev, stage1_oof, choice["cascade_recall_oof"]
            )
            baseline_metrics = evaluate_predictions(
                y_test, stage1_test, baseline_threshold
            )
            baseline_pred = (stage1_test >= baseline_threshold).astype(int)
            cascade_pred = (cascade_scores >= choice["threshold"]).astype(int)
            paired = paired_bootstrap_differences(
                y_test, baseline_pred, cascade_pred,
                n_bootstrap=BOOTSTRAPS,
                seed=seed + 3000 + target_index,
            )
            row = {
                "model": model_name,
                "recall_target": float(recall_floor),
                "cascade_oof_recall": float(choice["cascade_recall_oof"]),
                "baseline_oof_recall": float(baseline_oof_recall),
                "stage2_threshold": float(choice["threshold"]),
                "bayesian_precision_oof_mean": float(choice["precision_posterior_mean"]),
                "bayesian_precision_oof_lower_95": float(choice["precision_lower_credible_bound"]),
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
                "paired_bootstrap_differences": paired,
                "test_stage1_gate_rows": int(len(test_idx)),
            }
            model_result["recall_target_results"].append(row)
            matched_rows.append(row)
        model_results.append(model_result)

    report = {
        "experiment": "PyTorch MLP Stage 1 with XGBoost versus LightGBM Stage 2",
        "status": "research_only_not_promoted_to_deployed_inference",
        "test_rows": int(len(y_test)),
        "stage1": "Existing PyTorch MLP, unchanged; gate threshold selected on development OOF predictions for 99.5% recall.",
        "stage2": "XGBoost and LightGBM classifiers trained on Stage-1-positive development rows; five-fold cross-fitted Stage-2 predictions.",
        "threshold_selection": "Beta-Binomial precision posterior with Beta(1,1) prior; maximize the 95% posterior lower bound subject to 95%, 97%, or 98% end-to-end OOF recall.",
        "comparison": "Single-stage MLP threshold is selected on development OOF scores to match each tree cascade's achieved OOF recall. Both are compared on identical untouched test rows.",
        "bootstrap": "2,000 paired test-row bootstrap resamples; intervals are conditional on fitted models and frozen thresholds and exclude training/selection uncertainty.",
        "models": model_results,
        "warning": "Portfolio research only. The dataset is cross-sectional and this experiment is not clinically validated or for diagnosis, screening, treatment, or care decisions.",
    }
    (artifacts / "tree_stage2_cascade_comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    flat_rows = []
    for row in matched_rows:
        flat = {k: v for k, v in row.items() if k != "paired_bootstrap_differences"}
        for metric, values in row["paired_bootstrap_differences"].items():
            flat[f"{metric}_difference"] = values["difference_cascade_minus_baseline"]
            flat[f"{metric}_ci95_lower"] = values["paired_bootstrap_95_ci"][0]
            flat[f"{metric}_ci95_upper"] = values["paired_bootstrap_95_ci"][1]
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(
        artifacts / "tree_stage2_cascade_comparison.csv", index=False
    )
    pd.DataFrame(all_folds).to_csv(
        artifacts / "tree_stage2_cascade_folds.csv", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
