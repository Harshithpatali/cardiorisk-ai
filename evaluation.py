"""Calculate binary classification metrics at a frozen threshold."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


def expected_calibration_error(target, probabilities, bins=10):
    """Equal-width expected calibration error for descriptive calibration checks."""
    y = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    reliability = []
    for index in range(bins):
        upper_mask = scores < edges[index + 1] if index < bins - 1 else scores <= edges[index + 1]
        mask = (scores >= edges[index]) & upper_mask
        if not mask.any():
            continue
        mean_score = float(scores[mask].mean())
        observed_rate = float(y[mask].mean())
        error += float(mask.mean()) * abs(mean_score - observed_rate)
        reliability.append({
            "bin_lower": float(edges[index]),
            "bin_upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_score": mean_score,
            "observed_rate": observed_rate,
        })
    return float(error), reliability


def evaluate_predictions(target, probabilities, threshold):
    y_true = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    ece, reliability = expected_calibration_error(y_true, scores)
    return {
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "ece_10_bins": ece,
        "reliability_bins": reliability,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "specificity": float(matrix[0, 0] / max(matrix[0].sum(), 1)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }



def threshold_comparison(target, probabilities, thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)):
    """Compare fixed operating thresholds using development/OOF predictions."""
    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(target, probabilities, threshold)
        rows.append({
            key: metrics[key]
            for key in (
                "threshold", "precision", "recall", "specificity",
                "accuracy", "f1", "brier_score", "ece_10_bins",
            )
        })
    return rows
