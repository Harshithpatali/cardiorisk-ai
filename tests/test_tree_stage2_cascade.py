from tree_stage2_cascade import make_estimator


def test_xgboost_second_stage_estimator_configuration():
    model = make_estimator("xgboost", 17)
    assert model.get_params()["objective"] == "binary:logistic"
    assert model.get_params()["random_state"] == 17


def test_lightgbm_second_stage_estimator_configuration():
    model = make_estimator("lightgbm", 23)
    assert model.get_params()["objective"] == "binary"
    assert model.get_params()["random_state"] == 23
