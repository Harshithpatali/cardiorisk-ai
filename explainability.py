"""Integrated Gradients explanations for the CardioRisk AI research model."""

from pathlib import Path

import numpy as np
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
    """Create a small, reproducible reference sample using saved preprocessing."""
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


def explain_observation(
    model,
    background: torch.Tensor,
    transformed: np.ndarray,
    steps: int = 48,
):
    """Estimate Integrated Gradients for one observation on the model logit."""
    model.eval()
    observation = torch.tensor(transformed, dtype=torch.float32)
    baseline = background.mean(dim=0, keepdim=True)
    difference = observation - baseline
    accumulated_gradients = torch.zeros_like(observation)

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = (baseline + alpha * difference).detach()
        interpolated.requires_grad_(True)
        output = model(interpolated).sum()
        gradients = torch.autograd.grad(output, interpolated)[0]
        accumulated_gradients += gradients.detach()

    average_gradients = accumulated_gradients / steps
    attributions = difference * average_gradients
    return attributions.detach().cpu().numpy().reshape(-1)

def explanation_diagnostics(model, background: torch.Tensor, transformed: np.ndarray, attributions: np.ndarray):
    """Check the Integrated Gradients completeness residual for one row.

    A small residual is a numerical sanity check, not evidence that the
    explanation is causal, clinically meaningful, or stable across samples.
    """
    model.eval()
    observation = torch.as_tensor(transformed, dtype=torch.float32)
    baseline = background.mean(dim=0, keepdim=True)
    with torch.no_grad():
        output_delta = float((model(observation).sum() - model(baseline).sum()).item())
    attribution_sum = float(np.asarray(attributions, dtype=float).sum())
    residual = output_delta - attribution_sum
    return {
        "output_delta": output_delta,
        "attribution_sum": attribution_sum,
        "completeness_residual": float(residual),
        "absolute_residual": float(abs(residual)),
        "relative_residual": float(abs(residual) / max(abs(output_delta), 1e-8)),
    }
