"""Select a high-recall threshold from cross-fitted training predictions."""
import numpy as np
from sklearn.metrics import precision_recall_curve


def select_high_recall_threshold(target, probabilities, minimum_precision):
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    candidates = []
    for p_value, r_value, threshold in zip(precision[:-1], recall[:-1], thresholds):
        if p_value >= minimum_precision:
            candidates.append((float(r_value), float(p_value), float(threshold)))
    if not candidates:
        raise RuntimeError(
            f"No OOF threshold reaches precision >= {minimum_precision:.2f}"
        )
    best_recall = max(item[0] for item in candidates)
    tied = [item for item in candidates if np.isclose(item[0], best_recall)]
    best = max(tied, key=lambda item: (item[1], item[2]))
    metadata = {
        "method": "OOF maximum recall subject to OOF precision constraint",
        "minimum_precision": float(minimum_precision),
        "oof_recall": best[0],
        "oof_precision": best[1],
    }
    return best[2], metadata
