"""SHAP-based local explanations for the CardioRisk AI research model."""

from pathlib import Path

import numpy as np
import shap
import torch

from preprocessing import clean_dataset, transform_frame


FEATURE_LABELS = {
    "age_years": "Age",
    "height": "Height",
    "weight": "Weight",
    "ap_hi": "Systolic pressure",
    "ap_lo": "Diastolic pressure",
    "bmi": "Body mass index",
    "pulse_pressure": "Pulse pressure",
    "map": "Mean arterial pressure",
    "smoke": "Smoking",
    "alco": "Alcohol intake",
    "active": "Physical activity",
}


def build_background(dataset_path: Path, config: dict, sample_size: int = 64):
    """Create a small, reproducible background sample using saved preprocessing."""
    cleaned = clean_dataset(dataset_path)
    feature_frame = cleaned.drop(columns=["cardio"])
    sample = feature_frame.sample(
        n=min(sample_size, len(feature_frame)),
        random_state=42,
    )
    transformed = transform_frame(sample, config)
    return torch.tensor(transformed, dtype=torch.float32)


def feature_names(config: dict):
    """Return display names in the exact encoded column order."""
    names = [FEATURE_LABELS[column] for column in config["numeric_columns"]]

    category_labels = {
        "gender": {1: "Female", 2: "Male"},
        "cholesterol": {
            1: "Normal cholesterol",
            2: "Above-normal cholesterol",
            3: "High cholesterol",
        },
        "gluc": {
            1: "Normal glucose",
            2: "Above-normal glucose",
            3: "High glucose",
        },
    }

    for column, categories in zip(
        config["categorical_columns"],
        config["categories"],
    ):
        for category in categories:
            value = category.item() if hasattr(category, "item") else category
            label = category_labels.get(column, {}).get(value, str(value))
            names.append(f"{column.title()}: {label}")

    names.extend(FEATURE_LABELS[column] for column in config["binary_columns"])
    return names


def explain_observation(model, background: torch.Tensor, transformed: np.ndarray):
    """Calculate Gradient SHAP values for one observation on the model logit."""
    explainer = shap.GradientExplainer(model, background)
    observation = torch.tensor(transformed, dtype=torch.float32)
    values = explainer.shap_values(observation)

    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)
    values = np.squeeze(values)

    if values.ndim != 1:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    return values
