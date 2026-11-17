import numpy as np

from specificity_investigation import (
    fold_interval, metrics_at_threshold, summarize_candidate,
)


def test_metrics_at_threshold_calculates_specificity_and_recall():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9])
    result = metrics_at_threshold(y, p, 0.5)
    assert result["specificity"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert result["tn"] == 2
    assert result["fp"] == 1


def test_fold_interval_reports_mean_and_ordered_bounds():
    result = fold_interval([0.5, 0.6, 0.7, 0.8, 0.9])
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_equal_recall_summary_meets_requested_floor():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    folds = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    rows = summarize_candidate("toy", y, p, folds)
    assert len(rows) == 3
    assert all(row["recall"] + 1e-12 >= row["recall_floor"] for row in rows)
    assert all("specificity_fold_ci95" in row for row in rows)
