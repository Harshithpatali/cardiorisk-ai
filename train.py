"""Leakage-aware orchestration for the cross-fitted PyTorch MLP."""
import numpy as np

from config import (
    ARTIFACT_DIR, BATCH_SIZE, EARLY_STOPPING_PATIENCE, MAX_EPOCHS,
    MIN_OOF_PRECISION, N_SPLITS, RANDOM_SEED, TEST_SIZE, DATA_PATH,
)
from cross_validation import run_cross_validation
from data_split import split_holdout
from evaluation import evaluate_predictions
from model import CardioMLP
from model_training import predict_probabilities, train_fixed_epochs
from preprocessing import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
    clean_dataset, fit_preprocessor, transform_frame,
)
from save_artifacts import save_artifacts
from threshold_selection import select_high_recall_threshold


def main():
    data = clean_dataset(DATA_PATH)
    features = data[FEATURE_COLUMNS].reset_index(drop=True)
    target = data["cardio"].astype(int).to_numpy()

    # Split once before any learned preprocessing or model selection.
    x_train, x_test, y_train, y_test = split_holdout(
        features, target, TEST_SIZE, RANDOM_SEED
    )

    oof, fold_reports, selected_epochs, histories = run_cross_validation(
        x_train,
        y_train,
        N_SPLITS,
        RANDOM_SEED,
        BATCH_SIZE,
        MAX_EPOCHS,
        EARLY_STOPPING_PATIENCE,
    )
    if np.isnan(oof).any():
        raise RuntimeError("Cross-validation did not generate complete OOF predictions.")

    threshold, threshold_info = select_high_recall_threshold(
        y_train, oof, MIN_OOF_PRECISION
    )
    oof_metrics = evaluate_predictions(y_train, oof, threshold)
    final_epochs = max(1, int(np.median(selected_epochs)))

    # Refit preprocessing and model using development data only.
    final_config = fit_preprocessor(x_train)
    x_train_final = transform_frame(x_train, final_config)
    x_test_final = transform_frame(x_test, final_config)
    final_model = train_fixed_epochs(
        x_train_final, y_train, final_epochs, RANDOM_SEED, BATCH_SIZE
    )
    test_probabilities = predict_probabilities(final_model, x_test_final)
    test_metrics = evaluate_predictions(y_test, test_probabilities, threshold)

    final_config.update({
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "binary_columns": BINARY_COLUMNS,
        "input_dim": int(x_train_final.shape[1]),
    })
    report = {
        "model": "PyTorch MLP (128-64-32)",
        "evaluation_design": {
            "test_holdout_fraction": TEST_SIZE,
            "cv": f"{N_SPLITS}-fold StratifiedKFold on development partition",
            "preprocessing": "Imputation, one-hot encoding and scaling fitted separately inside each training fold; final preprocessor fitted on development rows only.",
            "threshold_selection": "Out-of-fold development predictions only; final test partition not used.",
            "epoch_selection": "Inner stratified validation split within each CV training fold; median selected epoch count used for final refit.",
            "test_policy": "Single final evaluation after model, preprocessing, epoch count and threshold are frozen.",
        },
        "default_operating_point": "high_recall",
        "threshold": float(threshold),
        "threshold_selection": threshold_info,
        "cross_validation": {
            "n_splits": N_SPLITS,
            "folds": fold_reports,
            "selected_epochs_by_fold": selected_epochs,
            "final_refit_epochs": final_epochs,
            "oof_training_metrics": oof_metrics,
            "oof_rows": int(len(oof)),
        },
        "test": test_metrics,
        "operating_points": {
            "high_recall": {
                "threshold": float(threshold),
                "selection": threshold_info,
                "oof": oof_metrics,
                "test": test_metrics,
            }
        },
        "rows_after_cleaning": int(len(data)),
        "split_sizes": {
            "train": int(len(x_train)),
            "test": int(len(x_test)),
            "cv_folds": N_SPLITS,
        },
        "warning": "Research/educational dataset metrics only; not clinical validation or individual medical risk.",
    }
    save_artifacts(final_model, final_config, report, histories, ARTIFACT_DIR)
    print("OOF development metrics:", oof_metrics)
    print("Held-out test metrics:", test_metrics)


if __name__ == "__main__":
    main()
