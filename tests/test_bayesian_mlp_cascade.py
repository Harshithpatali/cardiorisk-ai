import numpy as np
import torch

from bayesian_mlp_cascade import (
    bayesian_precision_threshold,
    mc_dropout_probabilities,
    select_recall_threshold,
)
from model import CardioMLP


def test_stage1_threshold_meets_recall_floor():
    y = np.array([0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    threshold, recall = select_recall_threshold(y, scores, 0.75)
    assert threshold == 0.6
    assert recall >= 0.75


def test_bayesian_precision_threshold_returns_posterior_interval():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.95, 0.85, 0.75, 0.9, 0.7, 0.4, 0.3, 0.1])
    result = bayesian_precision_threshold(
        y_true=y,
        scores=scores,
        stage1_true_positives=3,
        total_positives=4,
        recall_floor=0.5,
    )
    lower, upper = result["precision_credible_interval_95"]
    assert 0.0 <= lower <= upper <= 1.0
    assert result["precision_lower_credible_bound"] == lower
    assert result["cascade_recall_oof"] >= 0.5


def test_mc_dropout_returns_mean_and_uncertainty():
    torch.manual_seed(7)
    model = CardioMLP(input_dim=5, dropout=0.4)
    x = np.random.default_rng(7).normal(size=(12, 5)).astype(np.float32)
    mean, std = mc_dropout_probabilities(model, x, passes=5, seed=7)
    assert mean.shape == (12,)
    assert std.shape == (12,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert ((mean >= 0.0) & (mean <= 1.0)).all()
    assert (std >= 0.0).all()
