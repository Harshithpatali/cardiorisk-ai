import numpy as np

from boosting_calibration_study import expected_calibration_error, mean_ci


def test_ece_is_zero_for_perfectly_calibrated_groups():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    ece, bins = expected_calibration_error(y, p)
    assert ece == 0.0
    assert len(bins) == 2


def test_mean_ci_contains_mean():
    result = mean_ci([0.4, 0.5, 0.6, 0.7, 0.8])
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]
