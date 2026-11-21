"""Two-stage MLP cascade with Bayesian precision-threshold selection.

Stage 1 is the existing high-recall MLP. Stage 2 is a precision-oriented
PyTorch MLP trained only on Stage-1-positive development rows. Stage-2
thresholds are selected from cross-fitted development predictions using a
Beta-Binomial posterior lower credible bound for precision.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACT_DIR, BATCH_SIZE, DATA_PATH, RANDOM_SEED, TEST_SIZE
from data_split import split_holdout
from evaluation import evaluate_predictions
from model_training import predict_probabilities, train_fixed_epochs
from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame


RECALL_FLOORS = (0.95, 0.97, 0.98)
STAGE2_FOLDS = 5
MC_DROPOUT_PASSES = 30
BAYESIAN_PRIOR_ALPHA = 1.0
BAYESIAN_PRIOR_BETA = 1.0
CREDIBLE_LEVEL = 0.95


def select_recall_threshold(y_true, scores, recall_floor):
    """Choose the highest Stage-1 threshold that reaches the requested recall."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    candidates = np.unique(p)
    feasible = []
    for threshold in candidates:
        predicted = p >= threshold
        recall = float(np.sum(predicted & (y == 1)) / max(np.sum(y == 1), 1))
        if recall >= recall_floor:
            feasible.append((float(threshold), recall))
    if not feasible:
        raise RuntimeError(f"Stage 1 cannot reach recall floor {recall_floor:.0%}")
    return max(feasible, key=lambda item: item[0])


def mc_dropout_probabilities(model, features, passes=MC_DROPOUT_PASSES, seed=RANDOM_SEED):
    """Approximate a Bayesian predictive distribution with MC dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    samples = []
    with torch.no_grad():
        for pass_index in range(passes):
            torch.manual_seed(seed + pass_index)
            samples.append(torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1))
    model.eval()
    draws = np.vstack(samples)
    return draws.mean(axis=0), draws.std(axis=0, ddof=1)


def bayesian_precision_threshold(y_true, scores, stage1_true_positives,
                                 total_positives, recall_floor):
    """Maximize the 95% posterior lower bound for precision under a cascade
    recall constraint. Precision has a Beta(1,1) prior and a binomial
    likelihood, yielding Beta(TP+1, FP+1) posterior distributions.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0.0, 1.0, min(401, len(p)))))
    feasible = []
    for threshold in candidates:
        pred = p >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        cascade_recall = (
            float(stage1_true_positives * (tp / max(int(np.sum(y == 1)), 1)))
            / max(total_positives, 1)
        )
        if cascade_recall + 1e-12 < recall_floor:
            continue
        a = BAYESIAN_PRIOR_ALPHA + tp
        b = BAYESIAN_PRIOR_BETA + fp
        lower = float(beta.ppf((1.0 - CREDIBLE_LEVEL) / 2.0, a, b))
        upper = float(beta.ppf(1.0 - (1.0 - CREDIBLE_LEVEL) / 2.0, a, b))
        mean = float(a / (a + b))
        feasible.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "cascade_recall_oof": cascade_recall,
            "precision_posterior_mean": mean,
            "precision_credible_interval_95": [lower, upper],
            "precision_lower_credible_bound": lower,
        })
    if not feasible:
        raise RuntimeError(
            f"No Stage-2 threshold satisfies the {recall_floor:.0%} cascade recall floor"
        )
    # Conservative Bayesian decision: maximize the lower credible bound;
    # tie-break on higher cascade recall, then higher posterior mean.
    return max(
        feasible,
        key=lambda row: (
            row["precision_lower_credible_bound"],
            row["cascade_recall_oof"],
            row["precision_posterior_mean"],
        ),
    )


def stage2_oof_predictions(features, target, epochs, seed):
    splitter = StratifiedKFold(n_splits=STAGE2_FOLDS, shuffle=True, random_state=seed)
    oof_mean = np.full(len(target), np.nan, dtype=float)
    oof_std = np.full(len(target), np.nan, dtype=float)
    fold_rows = []
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(features, target), start=1):
        x_fit = features.iloc[fit_idx].reset_index(drop=True)
        x_hold = features.iloc[hold_idx].reset_index(drop=True)
        y_fit = target[fit_idx]
        config = fit_preprocessor(x_fit)
        fit_matrix = transform_frame(x_fit, config)
        hold_matrix = transform_frame(x_hold, config)
        model = train_fixed_epochs(
            fit_matrix, y_fit, epochs, seed + 100 + fold, BATCH_SIZE,
            dropout=0.35, positive_weight_multiplier=1.0,
        )
        mean, std = mc_dropout_probabilities(
            model, hold_matrix, seed=seed + 1000 + fold
        )
        oof_mean[hold_idx] = mean
        oof_std[hold_idx] = std
        fold_rows.append({
            "fold": fold,
            "rows": int(len(hold_idx)),
            "stage2_training_rows": int(len(fit_idx)),
            "mc_dropout_mean_std": float(np.mean(std)),
        })
    if np.isnan(oof_mean).any() or np.isnan(oof_std).any():
        raise RuntimeError("Stage-2 cross-validation did not produce complete predictions")
    return oof_mean, oof_std, fold_rows


def run():
    artifact_dir = Path(ARTIFACT_DIR)
    baseline_path = artifact_dir / "metrics.json"
    oof_path = artifact_dir / "mlp_oof_predictions.csv"
    if not baseline_path.exists() or not oof_path.exists():
        raise FileNotFoundError("Run train.py first to create baseline metrics and MLP OOF predictions.")

    baseline = json.loads(baseline_path.read_text())
    baseline_oof = pd.read_csv(oof_path)
    data = clean_dataset(DATA_PATH)
    features = data[FEATURE_COLUMNS].reset_index(drop=True)
    target = data["cardio"].astype(int).to_numpy()
    x_train, x_test, y_train, y_test = split_holdout(features, target, TEST_SIZE, RANDOM_SEED)

    if len(baseline_oof) != len(y_train) or not np.array_equal(
        baseline_oof["target"].to_numpy(dtype=int), y_train
    ):
        raise RuntimeError("Saved OOF predictions do not match the current deterministic development split.")
    stage1_oof = baseline_oof["oof_probability"].to_numpy(dtype=float)
    final_epochs = int(baseline["cross_validation"]["final_refit_epochs"])

    # Refit the unchanged Stage-1 architecture on development rows only.
    stage1_config = fit_preprocessor(x_train)
    x_train_matrix = transform_frame(x_train, stage1_config)
    x_test_matrix = transform_frame(x_test, stage1_config)
    stage1_model = train_fixed_epochs(
        x_train_matrix, y_train, final_epochs, RANDOM_SEED, BATCH_SIZE
    )
    stage1_test = predict_probabilities(stage1_model, x_test_matrix)

    total_test_positives = int(np.sum(y_test == 1))
    total_dev_positives = int(np.sum(y_train == 1))
    experiments = []
    all_fold_rows = []
    for floor_index, recall_floor in enumerate(RECALL_FLOORS):
        # Keep Stage 1 deliberately more sensitive than the final cascade
        # target. This recall buffer gives Stage 2 room to reject false
        # positives while retaining the requested end-to-end recall.
        stage1_gate_recall_target = 0.995
        stage1_threshold, stage1_recall = select_recall_threshold(
            y_train, stage1_oof, stage1_gate_recall_target
        )
        dev_gate = stage1_oof >= stage1_threshold
        test_gate = stage1_test >= stage1_threshold
        dev_indices = np.flatnonzero(dev_gate)
        test_indices = np.flatnonzero(test_gate)
        if len(dev_indices) < STAGE2_FOLDS * 2:
            raise RuntimeError("Too few Stage-1-positive rows for Stage-2 cross-validation.")

        stage2_x = x_train.iloc[dev_indices].reset_index(drop=True)
        stage2_y = y_train[dev_indices]
        stage2_oof, stage2_oof_std, fold_rows = stage2_oof_predictions(
            stage2_x, stage2_y, final_epochs, RANDOM_SEED + floor_index * 10000
        )
        for row in fold_rows:
            all_fold_rows.append({"cascade_recall_target": recall_floor, **row})

        stage1_dev_tp = int(np.sum(dev_gate & (y_train == 1)))
        bayes_choice = bayesian_precision_threshold(
            stage2_y, stage2_oof, stage1_dev_tp, total_dev_positives, recall_floor
        )

        # Refit Stage 2 on every Stage-1-positive development row.
        final_stage2_config = fit_preprocessor(stage2_x)
        stage2_train_matrix = transform_frame(stage2_x, final_stage2_config)
        stage2_model = train_fixed_epochs(
            stage2_train_matrix, stage2_y, final_epochs,
            RANDOM_SEED + 50000 + floor_index, BATCH_SIZE,
            dropout=0.35, positive_weight_multiplier=1.0,
        )
        if len(test_indices):
            stage2_test_matrix = transform_frame(
                x_test.iloc[test_indices].reset_index(drop=True), final_stage2_config
            )
            stage2_test_mean, stage2_test_std = mc_dropout_probabilities(
                stage2_model, stage2_test_matrix,
                seed=RANDOM_SEED + 60000 + floor_index,
            )
        else:
            stage2_test_mean = np.array([], dtype=float)
            stage2_test_std = np.array([], dtype=float)

        cascade_scores = np.zeros(len(y_test), dtype=float)
        cascade_scores[test_indices] = stage2_test_mean
        cascade_metrics = evaluate_predictions(
            y_test, cascade_scores, bayes_choice["threshold"]
        )
        stage1_metrics = evaluate_predictions(y_test, stage1_test, stage1_threshold)
        stage1_test_tp = int(np.sum(test_gate & (y_test == 1)))
        stage2_test_tp = int(np.sum(
            (stage2_test_mean >= bayes_choice["threshold"]) & (y_test[test_indices] == 1)
        ))
        cascade_metrics["stage1_gate_rows"] = int(len(test_indices))
        cascade_metrics["stage1_gate_fraction"] = float(len(test_indices) / len(y_test))
        cascade_metrics["stage2_mc_dropout_mean_uncertainty"] = (
            float(np.mean(stage2_test_std)) if len(stage2_test_std) else None
        )
        cascade_metrics["stage1_false_negatives"] = int(np.sum((~test_gate) & (y_test == 1)))
        cascade_metrics["stage2_false_negatives_among_stage1_positives"] = int(
            np.sum(test_gate & (y_test == 1)) - stage2_test_tp
        )
        cascade_metrics["bayesian_precision_posterior"] = {
            "prior": {"alpha": BAYESIAN_PRIOR_ALPHA, "beta": BAYESIAN_PRIOR_BETA},
            "credible_level": CREDIBLE_LEVEL,
            "oof_posterior_mean": bayes_choice["precision_posterior_mean"],
            "oof_95_percent_credible_interval": bayes_choice["precision_credible_interval_95"],
            "oof_lower_credible_bound": bayes_choice["precision_lower_credible_bound"],
        }
        experiments.append({
            "cascade_recall_target": recall_floor,
            "stage1_gate_recall_target": stage1_gate_recall_target,
            "stage1_threshold": stage1_threshold,
            "stage1_oof_recall": stage1_recall,
            "stage1_test_metrics": stage1_metrics,
            "stage2_training_rows": int(len(dev_indices)),
            "stage2_training_positive_rate": float(np.mean(stage2_y)),
            "stage2_oof_mean_mc_uncertainty": float(np.mean(stage2_oof_std)),
            "bayesian_stage2_threshold_selection": bayes_choice,
            "cascade_test_metrics": cascade_metrics,
            "test_stage1_positive_rows": int(len(test_indices)),
            "test_stage1_true_positives": stage1_test_tp,
        })

    report = {
        "experiment": "Two-stage PyTorch MLP cascade with Bayesian precision decision",
        "status": "research_experiment_only_not_promoted_to_deployed_inference",
        "method": {
            "stage1": "Existing PyTorch MLP; gate threshold targets 99.5% development OOF recall to create a recall buffer for Stage 2.",
            "stage2": "Second PyTorch MLP trained only on Stage-1-positive OOF development rows; precision-oriented unweighted BCE objective.",
            "bayesian_component": "Beta-Binomial posterior for Stage-2 precision with Beta(1,1) prior. Threshold maximizes the 95% posterior lower credible bound subject to the requested end-to-end cascade OOF recall target (95%, 97%, or 98%).",
            "uncertainty": "Monte Carlo dropout (30 stochastic passes) estimates an approximate predictive mean and epistemic uncertainty for Stage 2.",
            "leakage_controls": "Stage-1 OOF scores determine development gating; Stage-2 predictions are cross-fitted on gated development rows; all thresholds are selected without the untouched test set.",
            "limitation": "Bayesian credible intervals quantify binomial uncertainty conditional on this dataset and selected threshold; they do not establish clinical validity or population generalisability.",
        },
        "baseline_mlp_test_metrics": baseline["test"],
        "recall_floors": list(RECALL_FLOORS),
        "stage2_cv_folds": STAGE2_FOLDS,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "experiments": experiments,
        "test_rows": int(len(y_test)),
        "warning": "Portfolio research only. Not clinically validated and not for diagnosis, screening, treatment, or care decisions.",
    }
    (artifact_dir / "mlp_bayesian_cascade.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(all_fold_rows).to_csv(
        artifact_dir / "mlp_bayesian_cascade_folds.csv", index=False
    )
    pd.DataFrame([
        {
            "cascade_recall_target": row["cascade_recall_target"],
            "stage1_gate_test_recall": row["stage1_test_metrics"]["recall"],
            "stage1_test_precision": row["stage1_test_metrics"]["precision"],
            "cascade_test_recall": row["cascade_test_metrics"]["recall"],
            "cascade_test_precision": row["cascade_test_metrics"]["precision"],
            "cascade_test_specificity": row["cascade_test_metrics"]["specificity"],
            "cascade_test_f1": row["cascade_test_metrics"]["f1"],
            "cascade_test_roc_auc": row["cascade_test_metrics"]["roc_auc"],
            "cascade_test_false_positives": row["cascade_test_metrics"]["confusion_matrix"][0][1],
            "cascade_test_false_negatives": row["cascade_test_metrics"]["confusion_matrix"][1][0],
            "stage2_threshold": row["bayesian_stage2_threshold_selection"]["threshold"],
            "bayesian_precision_lower_95": row["bayesian_stage2_threshold_selection"]["precision_lower_credible_bound"],
            "bayesian_precision_mean": row["bayesian_stage2_threshold_selection"]["precision_posterior_mean"],
        }
        for row in experiments
    ]).to_csv(artifact_dir / "mlp_bayesian_cascade_summary.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
