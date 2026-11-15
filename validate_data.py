"""Validate the source CSV before any modelling task."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
}


def validate_dataset(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    frame = pd.read_csv(source, sep=";")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(frame) < 1000:
        raise ValueError(f"Dataset is too small: {len(frame)} rows")
    targets = set(frame["cardio"].dropna().unique())
    if not targets.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, found: {targets}")
    if frame["cardio"].isna().any():
        raise ValueError("Target column contains missing values")
    if frame["cardio"].nunique() != 2:
        raise ValueError("Both target classes must be present")
    return frame


def main():
    from config import DATA_PATH
    frame = validate_dataset(DATA_PATH)
    print(f"Dataset valid: {len(frame):,} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
