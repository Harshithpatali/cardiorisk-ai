"""Leakage-safe five-fold cross-fitted MLP training.

The final test partition is split off before CV and is never used for
preprocessing, epoch selection, threshold selection, or model selection.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import CardioMLP
from preprocessing import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
    clean_dataset, fit_preprocessor, transform_frame,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cardio_train.csv"
OUT = ROOT / "artifacts"
SEED = 42
N_SPLITS = 5
MAX_EPOCHS = 45
PATIENCE = 7
BATCH_SIZE = 512
MIN_VALIDATION_PRECISION = 0.55


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32)
        return torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1)


def evaluate(y, probabilities, threshold):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, pred)),
        "specificity": float(cm[0, 0] / max(cm[0].sum(), 1)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }


def fit_mlp(x_train, y_train, x_valid, y_valid, max_epochs=MAX_EPOCHS):
    """Train with early stopping on a validation set; returns best epoch/state."""
    seed_all()
    model = CardioMLP(x_train.shape[1])
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    pos_weight = torch.tensor([negatives / positives * 1.3], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    valid_ds = TensorDataset(
        torch.as_tensor(x_valid, dtype=torch.float32),
        torch.as_tensor(y_valid, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=(len(train_ds) % BATCH_SIZE == 1), generator=generator,
    )
    valid_loader = DataLoader(valid_ds, batch_size=2048, shuffle=False)

    best_loss = float("inf")
    best_state = None
    best_epoch = 1
    stale = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += loss.item() * len(xb)

        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for xb, yb in valid_loader:
                valid_total += criterion(model(xb), yb).item() * len(xb)
        valid_loss = valid_total / len(valid_ds)
        train_loss = train_total / len(train_ds)
        scheduler.step(valid_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "validation_loss": float(valid_loss),
        })

        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("No valid MLP checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss, history


def fit_fixed_epochs(x_train, y_train, epochs):
    """Refit a fresh model on all permitted training rows for selected epochs."""
    seed_all()
    model = CardioMLP(x_train.shape[1])
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives * 1.3], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=(len(ds) % BATCH_SIZE == 1), generator=generator,
    )
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


def choose_high_recall_threshold(y, probabilities, min_precision=MIN_VALIDATION_PRECISION):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    eligible = [
        (float(r), float(t), float(p))
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if p >= min_precision
    ]
    if not eligible:
        raise RuntimeError(
            f"No out-of-fold threshold reaches precision >= {min_precision:.2f}."
        )
    # Highest recall; among ties choose the higher precision, then higher threshold.
    best_recall = max(item[0] for item in eligible)
    tied = [item for item in eligible if np.isclose(item[0], best_recall)]
    recall_value, threshold, precision_value = max(tied, key=lambda item: (item[2], item[1]))
    return threshold, {
        "method": "OOF maximum recall subject to OOF precision constraint",
        "minimum_precision": float(min_precision),
        "oof_precision_at_threshold": precision_value,
        "oof_recall_at_threshold": recall_value,
    }


def main():
    OUT.mkdir(exist_ok=True)
    seed_all()

    # Deterministic row-level cleaning/feature engineering only. No learned
    # statistic (imputation, category vocabulary, or scaling) is fit here.
    data = clean_dataset(DATA)
    x_all = data[FEATURE_COLUMNS].reset_index(drop=True)
    y_all = data["cardio"].astype(int).to_numpy()

    # Lock away the final test set before any CV, threshold or epoch selection.
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
    )
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probabilities = np.full(len(x_train), np.nan, dtype=np.float64)
    fold_reports = []
    fold_epochs = []
    fold_histories = []

    for fold, (fit_idx, hold_idx) in enumerate(cv.split(x_train, y_train), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        x_fold = x_train.iloc[fit_idx].reset_index(drop=True)
        y_fold = y_train[fit_idx]
        x_hold = x_train.iloc[hold_idx].reset_index(drop=True)
        y_hold = y_train[hold_idx]

        # Inner split determines epoch count. Its preprocessor is fitted only
        # on inner-fit rows, never on inner validation or the outer holdout.
        inner_fit_idx, inner_val_idx = train_test_split(
            np.arange(len(x_fold)), test_size=0.15, random_state=SEED + fold,
            stratify=y_fold,
        )
        x_inner_fit = x_fold.iloc[inner_fit_idx].reset_index(drop=True)
        x_inner_val = x_fold.iloc[inner_val_idx].reset_index(drop=True)
        y_inner_fit = y_fold[inner_fit_idx]
        y_inner_val = y_fold[inner_val_idx]

        inner_config = fit_preprocessor(x_inner_fit)
        inner_fit = transform_frame(x_inner_fit, inner_config)
        inner_val = transform_frame(x_inner_val, inner_config)
        _, best_epoch, best_loss, history = fit_mlp(
            inner_fit, y_inner_fit, inner_val, y_inner_val
        )
        fold_epochs.append(best_epoch)
        fold_histories.append({
            "fold": fold, "selected_epochs": best_epoch,
            "best_inner_validation_loss": float(best_loss),
            "epochs": history,
        })

        # Cross-fitted prediction: refit preprocessing on the complete outer
        # fold training rows, then train there and predict only its holdout.
        fold_config = fit_preprocessor(x_fold)
        x_fold_scaled = transform_frame(x_fold, fold_config)
        x_hold_scaled = transform_frame(x_hold, fold_config)
        fold_model = fit_fixed_epochs(x_fold_scaled, y_fold, best_epoch)
        oof_probabilities[hold_idx] = predict(fold_model, x_hold_scaled)
        fold_report = evaluate(y_hold, oof_probabilities[hold_idx], 0.5)
        fold_reports.append({
            "fold": fold, "rows": int(len(hold_idx)),
            "selected_epochs": int(best_epoch),
            "metrics_at_0_5": fold_report,
        })
        print(
            f"fold={fold} epochs={best_epoch} "
            f"ROC-AUC={fold_report['roc_auc']:.4f} "
            f"PR-AUC={fold_report['pr_auc']:.4f}"
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("OOF predictions are incomplete.")

    # Threshold is selected solely from cross-fitted training predictions.
    threshold, threshold_info = choose_high_recall_threshold(
        y_train, oof_probabilities
    )
    oof_metrics = evaluate(y_train, oof_probabilities, threshold)

    # Epoch count is selected from inner validation folds, not from test data.
    final_epochs = max(1, int(np.median(fold_epochs)))

    # Fit final preprocessing on all training rows, then refit final MLP.
    final_config = fit_preprocessor(x_train)
    x_train_final = transform_frame(x_train, final_config)
    x_test_final = transform_frame(x_test, final_config)
    final_model = fit_fixed_epochs(x_train_final, y_train, final_epochs)

    # The test set is touched once, after all choices are frozen.
    test_probabilities = predict(final_model, x_test_final)
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    final_config["feature_columns"] = FEATURE_COLUMNS
    final_config["numeric_columns"] = NUMERIC_COLUMNS
    final_config["categorical_columns"] = CATEGORICAL_COLUMNS
    final_config["binary_columns"] = BINARY_COLUMNS
    final_config["input_dim"] = int(x_train_final.shape[1])
    torch.save(final_model.state_dict(), OUT / "mlp_state_dict.pt")
    (OUT / "preprocessing.json").write_text(json.dumps(final_config, indent=2))

    report = {
        "model": "PyTorch MLP (128-64-32)",
        "evaluation_design": {
            "test_holdout_fraction": 0.20,
            "cv": f"{N_SPLITS}-fold StratifiedKFold on training partition",
            "preprocessing": "Imputation, one-hot encoding and scaling fitted separately inside each training fold; final preprocessor fitted on training partition only.",
            "threshold_selection": "Out-of-fold training predictions only; final test set not used.",
            "epoch_selection": "Inner stratified validation split within each CV training fold; median selected epoch count used for final refit.",
            "test_policy": "Single final evaluation after model, preprocessing, epoch count and threshold are frozen.",
        },
        "default_operating_point": "high_recall",
        "threshold": float(threshold),
        "threshold_selection": threshold_info,
        "cross_validation": {
            "n_splits": N_SPLITS,
            "folds": fold_reports,
            "selected_epochs_by_fold": fold_epochs,
            "final_refit_epochs": final_epochs,
            "oof_training_metrics": oof_metrics,
            "oof_rows": int(len(oof_probabilities)),
        },
        "test": test_metrics,
        "rows_after_cleaning": int(len(data)),
        "split_sizes": {
            "train": int(len(x_train)), "test": int(len(x_test)),
            "cv_folds": N_SPLITS,
        },
        "warning": "Research/educational dataset metrics only; not clinical validation or individual medical risk.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    flat_history = []
    for item in fold_histories:
        for epoch in item["epochs"]:
            flat_history.append({"fold": item["fold"], **epoch})
    pd.DataFrame(flat_history).to_csv(OUT / "training_history.csv", index=False)
    print("\nOOF training metrics:", json.dumps(oof_metrics, indent=2))
    print("\nUntouched held-out test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
