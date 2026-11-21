import numpy as np

from threshold_selection import select_high_recall_threshold


def test_threshold_selection_maximizes_recall_subject_to_precision_guardrail():
    target = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.90, 0.80, 0.70, 0.60, 0.60, 0.40, 0.30, 0.20])

    threshold, report = select_high_recall_threshold(target, probabilities, 0.50)

    assert threshold == 0.60
    assert report["oof_recall"] == 1.0
    assert report["oof_precision"] >= 0.50
    assert report["minimum_precision"] == 0.50
