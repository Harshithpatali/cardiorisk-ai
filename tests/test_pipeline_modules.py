"""Unit tests for independent pipeline components."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_split import split_holdout
from evaluation import evaluate_predictions
from threshold_selection import select_high_recall_threshold
from validate_data import validate_dataset


def test_validation_accepts_uploaded_dataset():
    dataset = validate_dataset(ROOT / "data" / "cardio_train.csv")
    assert len(dataset) >= 1000
    assert set(dataset["cardio"].unique()) == {0, 1}


def test_holdout_split_is_stratified_and_disjoint():
    features = pd.DataFrame({"value": np.arange(1000)})
    target = np.array([0] * 500 + [1] * 500)
    x_train, x_test, y_train, y_test = split_holdout(
        features,
        target,
        0.2,
        42,
    )
    assert len(x_train) + len(x_test) == len(features)
    assert set(x_train["value"]).isdisjoint(set(x_test["value"]))
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


def test_threshold_selection_uses_precision_constraint():
    target = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.4, 0.6, 0.3])
    threshold, report = select_high_recall_threshold(
        target,
        scores,
        0.5,
    )
    predictions = scores >= threshold
    precision = predictions[target == 1].sum() / max(predictions.sum(), 1)
    assert 0 <= threshold <= 1
    assert precision >= 0.5
    assert report["minimum_precision"] == 0.5


def test_evaluation_returns_complete_metrics():
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_predictions(target, scores, 0.5)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
