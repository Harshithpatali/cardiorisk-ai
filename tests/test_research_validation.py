import numpy as np
import pandas as pd

from research_validation import (
    bootstrap_auc_ci, classification_metrics, expected_calibration_error,
    mean_ci, subgroup_report, threshold_table,
)


def test_classification_metrics_include_balanced_accuracy_and_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = classification_metrics(y, p, 0.5)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_expected_calibration_error_reports_bins():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.15, 0.85, 0.95])
    ece, bins = expected_calibration_error(y, p, bins=5)
    assert ece >= 0
    assert sum(item["count"] for item in bins) == 4


def test_mean_ci_and_bootstrap_are_reproducible():
    ci = mean_ci([0.7, 0.8, 0.9])
    assert ci["ci95_low"] < ci["mean"] < ci["ci95_high"]
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    first = bootstrap_auc_ci(y, p, "roc_auc", iterations=30)
    second = bootstrap_auc_ci(y, p, "roc_auc", iterations=30)
    assert first == second
    assert first["low"] <= 1.0 <= first["high"]


def test_threshold_and_subgroup_reports_are_complete():
    y = np.array([0, 1] * 250)
    p = np.array([0.2, 0.8] * 250)
    thresholds = threshold_table(y, p)
    assert len(thresholds) == 11
    frame = pd.DataFrame({
        "age_years": np.tile([35, 45, 55, 65], 125),
        "gender": np.tile([1, 2], 250),
    })
    groups = subgroup_report(frame, y, p, 0.5)
    assert set(groups) == {"age_band", "gender"}
    assert groups["gender"]["female"]["n"] == 250


def test_operating_point_frontier_respects_recall_floor_and_maximizes_threshold():
    from operating_point_analysis import operating_point_for_recall_floor
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9])
    row = operating_point_for_recall_floor(y, p, 2 / 3)
    assert row["recall"] >= 2 / 3
    assert row["threshold"] == 0.5
    assert row["specificity"] == 1.0
