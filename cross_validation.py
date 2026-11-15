"""Generate one out-of-fold prediction per training row."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from preprocessing import fit_preprocessor, transform_frame
from model_training import (
    predict_probabilities,
    train_fixed_epochs,
    train_with_early_stopping,
)
from evaluation import evaluate_predictions


def run_cross_validation(
    features,
    target,
    n_splits,
    random_seed,
    batch_size,
    max_epochs,
    patience,
):
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    oof = np.full(len(features), np.nan, dtype=float)
    fold_reports = []
    selected_epochs = []
    histories = []
    for fold_number, (fit_indices, hold_indices) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        print(f"Starting fold {fold_number}/{n_splits}")
        fold_train = features.iloc[fit_indices].reset_index(drop=True)
        fold_holdout = features.iloc[hold_indices].reset_index(drop=True)
        fold_target = target[fit_indices]
        holdout_target = target[hold_indices]
        inner_train_indices, inner_valid_indices = train_test_split(
            np.arange(len(fold_train)),
            test_size=0.15,
            random_state=random_seed + fold_number,
            stratify=fold_target,
        )
        inner_train = fold_train.iloc[inner_train_indices].reset_index(drop=True)
        inner_valid = fold_train.iloc[inner_valid_indices].reset_index(drop=True)
        inner_train_target = fold_target[inner_train_indices]
        inner_valid_target = fold_target[inner_valid_indices]
        inner_config = fit_preprocessor(inner_train)
        inner_train_matrix = transform_frame(inner_train, inner_config)
        inner_valid_matrix = transform_frame(inner_valid, inner_config)
        _, best_epoch, best_loss, history = train_with_early_stopping(
            inner_train_matrix,
            inner_train_target,
            inner_valid_matrix,
            inner_valid_target,
            random_seed + fold_number,
            batch_size,
            max_epochs,
            patience,
        )
        selected_epochs.append(best_epoch)
        histories.extend([
            {"fold": fold_number, **item}
            for item in history
        ])
        fold_config = fit_preprocessor(fold_train)
        fold_train_matrix = transform_frame(fold_train, fold_config)
        fold_holdout_matrix = transform_frame(fold_holdout, fold_config)
        fold_model = train_fixed_epochs(
            fold_train_matrix,
            fold_target,
            best_epoch,
            random_seed + fold_number,
            batch_size,
        )
        fold_probabilities = predict_probabilities(
            fold_model,
            fold_holdout_matrix,
        )
        oof[hold_indices] = fold_probabilities
        fold_metrics = evaluate_predictions(
            holdout_target,
            fold_probabilities,
            0.5,
        )
        fold_reports.append({
            "fold": fold_number,
            "rows": int(len(hold_indices)),
            "selected_epochs": int(best_epoch),
            "best_inner_validation_loss": float(best_loss),
            "metrics_at_0_5": fold_metrics,
        })
        print(
            f"Fold {fold_number} ROC-AUC: "
            f"{fold_metrics['roc_auc']:.4f}"
        )
    if np.isnan(oof).any():
        raise RuntimeError("Cross-validation did not generate complete OOF predictions")
    return oof, fold_reports, selected_epochs, histories
