import numpy as np
from tree_stage2_cascade import make_estimator, choose_cascade_threshold


def test_xgboost_second_stage_estimator_configuration():
    model = make_estimator("xgboost", 17)
    assert model.get_params()["objective"] == "binary:logistic"
    assert model.get_params()["random_state"] == 17


def test_lightgbm_second_stage_estimator_configuration():
    model = make_estimator("lightgbm", 23)
    assert model.get_params()["objective"] == "binary"
    assert model.get_params()["random_state"] == 23


def test_stage2_threshold_counts_direct_and_borderline_predictions():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p1 = np.array([0.8, 0.9, 0.3, 0.4, 0.25, 0.1, 0.7, 0.2])
    p2 = np.array([0.0, 0.0, 0.9, 0.1, 0.8, 0.0, 0.0, 0.0])
    result = choose_cascade_threshold(y, p1, p2, gate=0.2, recall_floor=0.5)
    assert result["direct_positive_rows"] == 3
    assert result["routed_rows"] == 4
    assert 0 <= result["threshold"] <= 1


def test_cascade_predictions_align_full_test_rows():
    from tree_stage2_cascade import build_cascade_predictions

    direct = np.array([True, False, False, False, True])
    routed = np.array([False, True, True, False, False])
    full_scores = np.array([0.0, 0.8, 0.2, 0.0, 0.0])
    predictions = build_cascade_predictions(direct, routed, full_scores, 0.5)
    np.testing.assert_array_equal(predictions, np.array([1, 1, 0, 0, 1]))


def test_cascade_predictions_reject_misaligned_arrays():
    import pytest
    from tree_stage2_cascade import build_cascade_predictions

    with pytest.raises(ValueError, match="align"):
        build_cascade_predictions(
            np.array([True, False, False]),
            np.array([False, True, False]),
            np.array([0.8, 0.2]),
            0.5,
        )


def test_cascade_predictions_reject_overlapping_routes():
    import pytest
    from tree_stage2_cascade import build_cascade_predictions

    with pytest.raises(ValueError, match="disjoint"):
        build_cascade_predictions(
            np.array([True, False]),
            np.array([True, True]),
            np.array([0.9, 0.8]),
            0.5,
        )
