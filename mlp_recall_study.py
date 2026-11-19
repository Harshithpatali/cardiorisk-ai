"""Development-only recall-focused MLP experiments; never touches the final holdout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACT_DIR, BATCH_SIZE, DATA_PATH, RANDOM_SEED
from data_split import split_holdout
from model_training import predict_probabilities, train_fixed_epochs
from preprocessing import (
    FEATURE_COLUMNS, RESEARCH_NUMERIC_COLUMNS, add_research_features,
    clean_dataset, fit_preprocessor, transform_frame,
)

FOLDS = 3
EPOCHS = 20
RECALL_FLOORS = (0.95, 0.97, 0.98)
EXPERIMENTS = {
    "baseline_adamw_pw1.3": {
        "optimizer_name": "adamw", "positive_weight_multiplier": 1.3,
        "dropout": 0.30, "weight_decay": 1e-4, "learning_rate": 1e-3,
        "engineered": False,
    },
    "adam_pw1.3": {
        "optimizer_name": "adam", "positive_weight_multiplier": 1.3,
        "dropout": 0.30, "weight_decay": 1e-4, "learning_rate": 1e-3,
        "engineered": False,
    },
    "sgd_momentum_pw1.3": {
        "optimizer_name": "sgd", "positive_weight_multiplier": 1.3,
        "dropout": 0.30, "weight_decay": 1e-4, "learning_rate": 1e-2,
        "engineered": False,
    },
    "adamw_pw1.7": {
        "optimizer_name": "adamw", "positive_weight_multiplier": 1.7,
        "dropout": 0.30, "weight_decay": 1e-4, "learning_rate": 1e-3,
        "engineered": False,
    },
    "adamw_pw2.0": {
        "optimizer_name": "adamw", "positive_weight_multiplier": 2.0,
        "dropout": 0.30, "weight_decay": 1e-4, "learning_rate": 1e-3,
        "engineered": False,
    },
    "strong_regularization": {
        "optimizer_name": "adamw", "positive_weight_multiplier": 1.3,
        "dropout": 0.45, "weight_decay": 1e-3, "learning_rate": 1e-3,
        "engineered": False,
    },
    "engineered_features": {
        "optimizer_name": "adamw", "positive_weight_multiplier": 1.3,
        "dropout": 0.30, "weight_decay": 1e-4, "learning_rate": 1e-3,
        "engineered": True,
    },
}


def summarize_operating_point(y, probabilities, recall_floor):
    precision_curve, recall_curve, thresholds = precision_recall_curve(y, probabilities)
    eligible = np.flatnonzero(recall_curve[:-1] + 1e-12 >= recall_floor)
    if not len(eligible):
        raise RuntimeError(f"No threshold meets recall floor {recall_floor}")
    threshold = float(thresholds[int(eligible[-1])])
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "recall_floor": float(recall_floor),
        "threshold": threshold,
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def main():
    data = clean_dataset(DATA_PATH)
    base_x = data[FEATURE_COLUMNS].reset_index(drop=True)
    y = data["cardio"].astype(int).to_numpy()
    x_dev, _, y_dev, _ = split_holdout(base_x, y, 0.20, RANDOM_SEED)
    engineered_dev = add_research_features(x_dev)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)

    results, all_fold_rows, all_oof_rows = [], [], []
    for name, params in EXPERIMENTS.items():
        use_engineered = params["engineered"]
        source_x = engineered_dev if use_engineered else x_dev
        extra = RESEARCH_NUMERIC_COLUMNS if use_engineered else ()
        oof = np.full(len(y_dev), np.nan, dtype=float)
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(source_x, y_dev), start=1):
            x_fit = source_x.iloc[fit_idx].reset_index(drop=True)
            x_val = source_x.iloc[val_idx].reset_index(drop=True)
            prep = fit_preprocessor(x_fit, extra_numeric_columns=extra)
            fit_matrix = transform_frame(x_fit, prep)
            val_matrix = transform_frame(x_val, prep)
            model = train_fixed_epochs(
                fit_matrix, y_dev[fit_idx], EPOCHS, RANDOM_SEED + fold,
                BATCH_SIZE,
                dropout=params["dropout"],
                positive_weight_multiplier=params["positive_weight_multiplier"],
                optimizer_name=params["optimizer_name"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )
            scores = predict_probabilities(model, val_matrix)
            oof[val_idx] = scores
            all_fold_rows.append({
                "experiment": name, "fold": fold,
                "roc_auc": float(roc_auc_score(y_dev[val_idx], scores)),
                "pr_auc": float(average_precision_score(y_dev[val_idx], scores)),
                "brier": float(brier_score_loss(y_dev[val_idx], scores)),
            })
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
        operating_points = [
            summarize_operating_point(y_dev, oof, floor) for floor in RECALL_FLOORS
        ]
        result = {
            "experiment": name,
            "optimizer": params["optimizer_name"],
            "positive_weight_multiplier": params["positive_weight_multiplier"],
            "dropout": params["dropout"],
            "weight_decay": params["weight_decay"],
            "learning_rate": params["learning_rate"],
            "engineered_features": bool(use_engineered),
            "epochs": EPOCHS,
            "oof_roc_auc": float(roc_auc_score(y_dev, oof)),
            "oof_pr_auc": float(average_precision_score(y_dev, oof)),
            "oof_brier": float(brier_score_loss(y_dev, oof)),
            "operating_points": operating_points,
            "scope": "three-fold development-only out-of-fold predictions",
        }
        results.append(result)
        all_oof_rows.extend({
            "experiment": name, "row_index": int(i), "target": int(y_dev[i]),
            "oof_probability": float(oof[i]),
        } for i in range(len(oof)))
        print(name, result["oof_roc_auc"], result["oof_pr_auc"], operating_points)

    payload = {
        "method": "Matched three-fold stratified OOF on the development partition; preprocessing fitted within each fold.",
        "development_rows": int(len(y_dev)),
        "epochs_per_fold": EPOCHS,
        "recall_floors": list(RECALL_FLOORS),
        "experiments": results,
        "fold_results": all_fold_rows,
        "selection_policy": "Compare specificity and precision at matched recall floors; do not select on recall alone. No automatic promotion.",
        "limitations": [
            "The untouched final holdout was not accessed by this experiment.",
            "Pooled OOF operating-point estimates are exploratory and may be optimistic after comparison across many configurations.",
            "A candidate must be frozen and evaluated once on a fresh independent cohort before any deployment decision.",
            "The dataset is cross-sectional and this project is educational research, not clinical validation or future-event prediction.",
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "mlp_recall_study.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(all_fold_rows).to_csv(ARTIFACT_DIR / "mlp_recall_study_folds.csv", index=False)
    pd.DataFrame(all_oof_rows).to_csv(ARTIFACT_DIR / "mlp_recall_study_oof.csv", index=False)


if __name__ == "__main__":
    main()
