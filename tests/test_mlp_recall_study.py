import numpy as np
import pandas as pd

from mlp_recall_study import summarize_operating_point
from preprocessing import RESEARCH_NUMERIC_COLUMNS, add_research_features, fit_preprocessor, transform_frame


def test_research_features_are_deterministic_and_finite():
    frame = pd.DataFrame([{
        "age_years": 50.0, "height": 170.0, "weight": 70.0,
        "ap_hi": 130.0, "ap_lo": 80.0, "bmi": 70.0 / 1.7**2,
        "pulse_pressure": 50.0, "map": 96.6667,
        "gender": 1, "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1,
    }])
    engineered = add_research_features(frame)
    assert all(column in engineered for column in RESEARCH_NUMERIC_COLUMNS)
    assert np.isfinite(engineered[RESEARCH_NUMERIC_COLUMNS].to_numpy()).all()
    config = fit_preprocessor(engineered, extra_numeric_columns=RESEARCH_NUMERIC_COLUMNS)
    matrix = transform_frame(engineered, config)
    assert matrix.shape == (1, config["input_dim"])


def test_matched_recall_operating_point_reports_false_negatives():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.6, 0.7, 0.9])
    result = summarize_operating_point(y, scores, 0.75)
    assert result["recall"] >= 0.75
    assert result["false_negatives"] == 1
    assert result["false_positives"] == 1
