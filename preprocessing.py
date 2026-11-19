"""Deterministic cleaning plus fold-fitted preprocessing for the cardio dataset."""
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure", "map"]
CATEGORICAL_COLUMNS = ["gender", "cholesterol", "gluc"]
BINARY_COLUMNS = ["smoke", "alco", "active"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS

RESEARCH_NUMERIC_COLUMNS = [
    "age_squared", "bmi_squared", "age_bmi_interaction",
    "age_systolic_interaction", "pulse_pressure_ratio", "systolic_map_ratio",
]


def clean_dataset(path):
    """Apply row-wise validity rules and deterministic feature engineering only."""
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        df = pd.read_csv(path, sep=";")
    required = {
        "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["cardio"]).copy()
    df["age_years"] = (df["age"] / 365.25).round(1)
    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / (height_m ** 2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map"] = df["ap_lo"] + df["pulse_pressure"] / 3.0

    bp_ok = (
        df["ap_hi"].between(70, 250)
        & df["ap_lo"].between(40, 180)
        & (df["ap_hi"] > df["ap_lo"])
    )
    body_ok = (
        df["height"].between(120, 220)
        & df["weight"].between(35, 200)
        & df["bmi"].between(12, 60)
    )
    target_ok = df["cardio"].isin([0, 1])
    cleaned = df.loc[bp_ok & body_ok & target_ok].drop(columns=["id", "age"]).copy()
    if cleaned.empty:
        raise ValueError("No rows remain after data-quality filters.")
    return cleaned.reset_index(drop=True)


def add_research_features(frame):
    """Add deterministic nonlinear/interaction features for experiments only."""
    result = frame.copy()
    age = result["age_years"].astype(float)
    bmi = result["bmi"].astype(float)
    systolic = result["ap_hi"].astype(float)
    map_value = result["map"].astype(float)
    result["age_squared"] = age ** 2
    result["bmi_squared"] = bmi ** 2
    result["age_bmi_interaction"] = age * bmi
    result["age_systolic_interaction"] = age * systolic
    result["pulse_pressure_ratio"] = result["pulse_pressure"] / result["ap_lo"].clip(lower=1)
    result["systolic_map_ratio"] = systolic / map_value.clip(lower=1)
    return result


def fit_preprocessor(frame, extra_numeric_columns=()):
    """Fit all learned preprocessing statistics on the supplied training rows."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_columns = NUMERIC_COLUMNS + list(extra_numeric_columns)
    feature_columns = numeric_columns + CATEGORICAL_COLUMNS + BINARY_COLUMNS
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    numeric_imputer = SimpleImputer(strategy="median")
    numeric_imputed = numeric_imputer.fit_transform(numeric)
    scaler = StandardScaler()
    scaler.fit(numeric_imputed)

    categorical = frame[CATEGORICAL_COLUMNS].to_numpy()
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    categorical_imputed = categorical_imputer.fit_transform(categorical)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit(categorical_imputed)

    binary = frame[BINARY_COLUMNS].to_numpy(dtype=np.float64)
    binary_imputer = SimpleImputer(strategy="most_frequent")
    binary_imputer.fit(binary)

    return {
        "numeric_imputer_statistics": numeric_imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "categorical_imputer_statistics": categorical_imputer.statistics_.tolist(),
        "categories": [
            [value.item() if hasattr(value, "item") else value for value in cats]
            for cats in encoder.categories_
        ],
        "binary_imputer_statistics": binary_imputer.statistics_.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "binary_columns": BINARY_COLUMNS,
        "feature_columns": feature_columns,
        "input_dim": int(
            len(numeric_columns)
            + sum(len(cats) for cats in encoder.categories_)
            + len(BINARY_COLUMNS)
        ),
    }


def transform_frame(frame, config):
    """Transform rows using only statistics/categories stored by fit_preprocessor."""
    x = frame[config["feature_columns"]].copy()

    numeric = x[config["numeric_columns"]].to_numpy(dtype=np.float64)
    numeric_stats = np.asarray(config["numeric_imputer_statistics"], dtype=np.float64)
    numeric = np.where(np.isnan(numeric), numeric_stats, numeric)
    mean = np.asarray(config["scaler_mean"], dtype=np.float64)
    scale = np.asarray(config["scaler_scale"], dtype=np.float64)
    numeric = (numeric - mean) / np.where(scale == 0, 1.0, scale)

    categorical = x[config["categorical_columns"]].to_numpy()
    cat_stats = np.asarray(config["categorical_imputer_statistics"])
    cat_blocks = []
    for index, (column, categories) in enumerate(
        zip(config["categorical_columns"], config["categories"])
    ):
        values = categorical[:, index].copy()
        missing = pd.isna(values)
        if missing.any():
            values[missing] = cat_stats[index]
        cats = np.asarray(categories)
        cat_blocks.append(np.column_stack([(values == category).astype(float) for category in cats]))

    binary = x[config["binary_columns"]].to_numpy(dtype=np.float64)
    binary_stats = np.asarray(config["binary_imputer_statistics"], dtype=np.float64)
    binary = np.where(np.isnan(binary), binary_stats, binary)

    result = np.concatenate([numeric, *cat_blocks, binary], axis=1).astype(np.float32)
    if result.shape[1] != int(config["input_dim"]):
        raise ValueError(
            f"Feature dimension mismatch: got {result.shape[1]}, expected {config['input_dim']}"
        )
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model input after preprocessing.")
    return result


def make_feature_row(age_years, gender, height, weight, ap_hi, ap_lo,
                     cholesterol, gluc, smoke, alco, active):
    pulse_pressure = float(ap_hi) - float(ap_lo)
    return pd.DataFrame([{
        "age_years": float(age_years),
        "height": float(height),
        "weight": float(weight),
        "ap_hi": float(ap_hi),
        "ap_lo": float(ap_lo),
        "bmi": float(weight) / (float(height) / 100.0) ** 2,
        "pulse_pressure": pulse_pressure,
        "map": float(ap_lo) + pulse_pressure / 3.0,
        "gender": int(gender),
        "cholesterol": int(cholesterol),
        "gluc": int(gluc),
        "smoke": int(smoke),
        "alco": int(alco),
        "active": int(active),
    }], columns=FEATURE_COLUMNS)
