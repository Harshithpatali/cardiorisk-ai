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
)


def evaluate_predictions(target, probabilities, threshold):
    y_true = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "specificity": float(matrix[0, 0] / max(matrix[0].sum(), 1)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }
