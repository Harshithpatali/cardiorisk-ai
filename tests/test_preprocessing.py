import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from preprocessing import (
    FEATURE_COLUMNS, fit_preprocessor, make_feature_row, transform_frame,
)


def test_preprocessing_is_fitted_only_on_supplied_training_rows():
    train = pd.DataFrame({
        "age_years": [40.0, 50.0, 60.0],
        "height": [160.0, 170.0, 180.0],
        "weight": [60.0, 70.0, 80.0],
        "ap_hi": [110.0, 120.0, 130.0],
        "ap_lo": [70.0, 80.0, 90.0],
        "bmi": [23.4, 24.2, 24.7],
        "pulse_pressure": [40.0, 40.0, 40.0],
        "map": [83.3, 93.3, 103.3],
        "gender": [1, 2, 1],
        "cholesterol": [1, 2, 1],
        "gluc": [1, 1, 2],
        "smoke": [0, 1, 0],
        "alco": [0, 0, 1],
        "active": [1, 1, 0],
    }, columns=FEATURE_COLUMNS)
    config = fit_preprocessor(train)
    # Extreme holdout values must not alter fitted training statistics.
    holdout = train.iloc[[0]].copy()
    holdout.loc[:, "age_years"] = 999.0
    transformed = transform_frame(holdout, config)
    assert config["scaler_mean"][0] == 50.0
    assert transformed.shape == (1, config["input_dim"])


def test_missing_values_are_imputed_from_training_statistics():
    train = pd.DataFrame({
        "age_years": [40.0, np.nan, 60.0],
        "height": [160.0, 170.0, 180.0],
        "weight": [60.0, 70.0, 80.0],
        "ap_hi": [110.0, 120.0, 130.0],
        "ap_lo": [70.0, 80.0, 90.0],
        "bmi": [23.4, 24.2, 24.7],
        "pulse_pressure": [40.0, 40.0, 40.0],
        "map": [83.3, 93.3, 103.3],
        "gender": [1, 2, 1],
        "cholesterol": [1, 2, 1],
        "gluc": [1, 1, 2],
        "smoke": [0, 1, 0],
        "alco": [0, 0, 1],
        "active": [1, 1, 0],
    }, columns=FEATURE_COLUMNS)
    config = fit_preprocessor(train)
    row = train.iloc[[1]]
    result = transform_frame(row, config)
    assert np.isfinite(result).all()


def test_dataset_cleaning_keeps_binary_target():
    from preprocessing import clean_dataset
    data = clean_dataset(ROOT / "data" / "cardio_train.csv")
    assert len(data) > 1000
    assert set(data.cardio.unique()) == {0, 1}


def test_saved_preprocessor_and_model_predict():
    import torch
    from model import CardioMLP

    cfg_path = ROOT / "artifacts" / "preprocessing.json"
    weights_path = ROOT / "artifacts" / "mlp_state_dict.pt"
    if not cfg_path.exists() or not weights_path.exists():
        # CI generates artifacts before tests; local tests may run before training.
        return
    cfg = json.loads(cfg_path.read_text())
    net = CardioMLP(cfg["input_dim"])
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()
    row = make_feature_row(55, 1, 165, 75, 120, 80, 1, 1, 0, 0, 1)
    x = transform_frame(row, cfg)
    with torch.no_grad():
        score = torch.sigmoid(net(torch.tensor(x, dtype=torch.float32))).item()
    assert 0 <= score <= 1
