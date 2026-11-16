"""Small, development-only cross-validated MLP architecture/loss experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from config import DATA_PATH, ARTIFACT_DIR, RANDOM_SEED, BATCH_SIZE
from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame
from model_training import train_fixed_epochs, predict_probabilities

VARIANTS = {
    "baseline_128_64_32_pw1.3": {"hidden_dims": (128, 64, 32), "dropout": 0.30, "positive_weight_multiplier": 1.3},
    "compact_64_32_pw1.0": {"hidden_dims": (64, 32), "dropout": 0.20, "positive_weight_multiplier": 1.0},
    "wide_256_128_64_pw1.0": {"hidden_dims": (256, 128, 64), "dropout": 0.40, "positive_weight_multiplier": 1.0},
}

def main():
    frame = clean_dataset(DATA_PATH)
    x, y = frame[FEATURE_COLUMNS].reset_index(drop=True), frame["cardio"].astype(int).to_numpy()
    from data_split import split_holdout
    x_dev, _, y_dev, _ = split_holdout(x, y, 0.2, RANDOM_SEED)
    primary = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    epochs = max(1, int(primary["cross_validation"]["final_refit_epochs"]))
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    results = []
    for name, params in VARIANTS.items():
        oof = np.full(len(x_dev), np.nan)
        fold_rows = []
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_dev, y_dev), start=1):
            x_fit, x_val = x_dev.iloc[fit_idx].reset_index(drop=True), x_dev.iloc[val_idx].reset_index(drop=True)
            prep = fit_preprocessor(x_fit)
            train_x, valid_x = transform_frame(x_fit, prep), transform_frame(x_val, prep)
            fitted = train_fixed_epochs(train_x, y_dev[fit_idx], epochs, RANDOM_SEED + fold,
                                        BATCH_SIZE, **params)
            scores = predict_probabilities(fitted, valid_x)
            oof[val_idx] = scores
            fold_rows.append({"fold": fold,
                "roc_auc": float(roc_auc_score(y_dev[val_idx], scores)),
                "pr_auc": float(average_precision_score(y_dev[val_idx], scores))})
        if np.isnan(oof).any():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
        results.append({"variant": name, "hidden_dims": list(params["hidden_dims"]),
            "dropout": params["dropout"], "positive_weight_multiplier": params["positive_weight_multiplier"],
            "epochs": epochs, "oof_roc_auc": float(roc_auc_score(y_dev, oof)),
            "oof_pr_auc": float(average_precision_score(y_dev, oof)), "folds": fold_rows,
            "scope": "three-fold development-only OOF experiment"})
        print(name, results[-1]["oof_roc_auc"], results[-1]["oof_pr_auc"])
    payload = {"method": "3-fold stratified development OOF; fold-specific preprocessing; fixed epoch count inherited from primary CV",
        "baseline": "baseline_128_64_32_pw1.3", "variants": results,
        "promotion_policy": "No variant is promoted automatically. Require repeatable gains and a fresh independent evaluation before deployment.",
        "limitations": ["Exploratory experiment; no final holdout evaluation was performed for these variants.",
            "Architecture comparisons use three folds and should not be treated as definitive model selection.",
            "The source dataset is cross-sectional and does not establish future disease risk."]}
    (ARTIFACT_DIR / "mlp_variant_benchmark.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame([{k:v for k,v in row.items() if k != "folds"} for row in results]).to_csv(
        ARTIFACT_DIR / "mlp_variant_benchmark.csv", index=False)

if __name__ == "__main__":
    main()
