"""Equal-recall specificity study: MLP class weights vs HistGradientBoosting.

All training and threshold selection use the development partition only.
No final holdout predictions are read or written by this experiment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACT_DIR, BATCH_SIZE, DATA_PATH, RANDOM_SEED, TEST_SIZE
from data_split import split_holdout
from model_training import predict_probabilities, train_fixed_epochs
from operating_point_analysis import operating_point_for_recall_floor
from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame

POSITIVE_WEIGHT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.3, 1.5)
RECALL_FLOORS = (0.95, 0.90, 0.85)


def fold_interval(values):
    """Student-t interval across held-out folds; descriptive with five folds."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def metrics_at_threshold(y, probabilities, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = float(recall_score(y, pred, zero_division=0))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "threshold": float(threshold),
        "recall": recall,
        "specificity": specificity,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def summarize_candidate(name, y, probabilities, fold_ids):
    rows = []
    for floor in RECALL_FLOORS:
        point = operating_point_for_recall_floor(y, probabilities, floor)
        threshold = point["threshold"]
        fold_specificity, fold_recall = [], []
        for fold in sorted(np.unique(fold_ids)):
            mask = np.asarray(fold_ids) == fold
            fold_metrics = metrics_at_threshold(
                np.asarray(y)[mask], np.asarray(probabilities)[mask], threshold
            )
            fold_specificity.append(fold_metrics["specificity"])
            fold_recall.append(fold_metrics["recall"])
        rows.append({
            "candidate": name,
            **point,
            "specificity_fold_ci95": fold_interval(fold_specificity),
            "recall_fold_ci95": fold_interval(fold_recall),
            "fold_specificity": [float(x) for x in fold_specificity],
            "fold_recall": [float(x) for x in fold_recall],
        })
    return rows


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    data = clean_dataset(DATA_PATH).reset_index(drop=True)
    features = data[FEATURE_COLUMNS].reset_index(drop=True)
    target = data["cardio"].astype(int).to_numpy()
    x_dev, _, y_dev, _ = split_holdout(features, target, TEST_SIZE, RANDOM_SEED)
    x_dev = x_dev.reset_index(drop=True)

    primary = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    fold_epochs = [
        int(row["selected_epochs"])
        for row in primary["cross_validation"]["folds"]
    ]
    if len(fold_epochs) != 5:
        raise RuntimeError("Expected five fold-specific epoch counts from primary CV")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_ids = np.zeros(len(x_dev), dtype=int)
    predictions = {}
    for multiplier in POSITIVE_WEIGHT_MULTIPLIERS:
        name = f"mlp_positive_weight_{multiplier:.2f}"
        predictions[name] = np.full(len(x_dev), np.nan, dtype=float)

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x_dev, y_dev), start=1):
        fold_ids[hold_idx] = fold
        x_fit = x_dev.iloc[fit_idx].reset_index(drop=True)
        x_hold = x_dev.iloc[hold_idx].reset_index(drop=True)
        prep = fit_preprocessor(x_fit)
        fit_matrix = transform_frame(x_fit, prep)
        hold_matrix = transform_frame(x_hold, prep)
        for multiplier in POSITIVE_WEIGHT_MULTIPLIERS:
            name = f"mlp_positive_weight_{multiplier:.2f}"
            fitted = train_fixed_epochs(
                fit_matrix, y_dev[fit_idx], fold_epochs[fold - 1],
                RANDOM_SEED + fold, BATCH_SIZE,
                positive_weight_multiplier=multiplier,
            )
            predictions[name][hold_idx] = predict_probabilities(fitted, hold_matrix)
        print(f"Completed class-weight fold {fold}/5")

    benchmark_path = ARTIFACT_DIR / "benchmark_oof_predictions.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError("Run research_validation.py before this study")
    benchmark = pd.read_csv(benchmark_path)
    if len(benchmark) != len(y_dev) or not np.array_equal(benchmark["target"], y_dev):
        raise RuntimeError("Benchmark OOF rows do not align with the primary development split")
    if not np.array_equal(benchmark["fold"].to_numpy(), fold_ids):
        raise RuntimeError("Benchmark fold assignments do not match this experiment")
    hgb_col = "hist_gradient_boosting__probability"
    predictions["hist_gradient_boosting"] = benchmark[hgb_col].to_numpy(dtype=float)

    all_rows = []
    prediction_frame = pd.DataFrame({"target": y_dev, "fold": fold_ids})
    for name, scores in predictions.items():
        if np.isnan(scores).any():
            raise RuntimeError(f"Incomplete OOF scores for {name}")
        prediction_frame[f"{name}__probability"] = scores
        all_rows.extend(summarize_candidate(name, y_dev, scores, fold_ids))

    payload = {
        "method": "5-fold stratified development OOF; fold-specific preprocessing; MLP fixed epochs inherited from primary fold CV",
        "positive_weight_multipliers": list(POSITIVE_WEIGHT_MULTIPLIERS),
        "recall_floors": list(RECALL_FLOORS),
        "candidates": all_rows,
        "selection_policy": "Compare specificity at matched recall. No automatic model or threshold promotion.",
        "scope": "development partition OOF predictions only; the final test holdout was not loaded or evaluated",
        "limitations": [
            "Five-fold internal validation only; fold confidence intervals are descriptive and do not establish external generalisation.",
            "Class-weight changes may alter both ranking and probability calibration.",
            "Thresholds are selected and evaluated on pooled development OOF predictions; independent confirmation is required.",
            "The dataset is cross-sectional and does not establish future cardiovascular events or clinical utility.",
        ],
    }
    (ARTIFACT_DIR / "specificity_investigation.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(all_rows).to_csv(ARTIFACT_DIR / "specificity_investigation.csv", index=False)
    prediction_frame.to_csv(ARTIFACT_DIR / "specificity_oof_predictions.csv", index=False)
    print(json.dumps({
        "candidates": len(predictions),
        "rows": len(all_rows),
        "recall_floors": list(RECALL_FLOORS),
        "best_specificity_by_floor": {
            str(floor): max(
                (row for row in all_rows if row["recall_floor"] == floor),
                key=lambda row: row["specificity"],
            )["candidate"]
            for floor in RECALL_FLOORS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
