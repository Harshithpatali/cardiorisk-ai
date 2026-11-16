"""Cross-validated model benchmark and statistical audit for CardioRisk AI.

This is a secondary research analysis. It uses only the development partition
for model comparison and does not overwrite the deployed PyTorch MLP or claim
clinical validity. Run after the primary training pipeline.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from preprocessing import FEATURE_COLUMNS, clean_dataset, fit_preprocessor, transform_frame

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cardio_train.csv"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
FOLDS = 5
BOOTSTRAPS = 500


def expected_calibration_error(y_true, probabilities, bins=10):
    """Equal-width-bin expected calibration error (descriptive, not a test)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    details = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not mask.any():
            continue
        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - observed)
        details.append({
            "bin_lower": float(edges[i]), "bin_upper": float(edges[i + 1]),
            "count": int(mask.sum()), "mean_score": confidence,
            "observed_rate": observed,
        })
    return float(ece), details


def classification_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / max(tn + fp, 1))
    ece, _ = expected_calibration_error(y, p)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_bins": ece,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=SEED,
        ),
        "logistic_sigmoid_calibrated": CalibratedClassifierCV(
            estimator=LogisticRegression(
                max_iter=1500, class_weight="balanced", solver="lbfgs"
            ),
            method="sigmoid", cv=3,
        ),
    }


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    half = float(t.ppf((1 + confidence) / 2, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def bootstrap_auc_ci(y, p, metric, seed=SEED, iterations=BOOTSTRAPS):
    """Percentile bootstrap interval over OOF rows; stratified by outcome."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    indices = [np.flatnonzero(y == label) for label in (0, 1)]
    values = []
    for _ in range(iterations):
        sampled = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in indices
        ])
        ys, ps = y[sampled], p[sampled]
        if metric == "roc_auc":
            values.append(roc_auc_score(ys, ps))
        elif metric == "pr_auc":
            values.append(average_precision_score(ys, ps))
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
    return {
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
        "method": "stratified percentile bootstrap over out-of-fold rows",
        "iterations": iterations,
    }


def threshold_table(y, p):
    rows = []
    for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        m = classification_metrics(y, p, threshold)
        rows.append({k: m[k] for k in (
            "threshold", "precision", "recall", "specificity",
            "balanced_accuracy", "f1", "accuracy"
        )})
    return rows


def subgroup_report(frame, y, p, threshold):
    """Descriptive OOF subgroup metrics; small groups are flagged, not ranked."""
    report = {}
    groups = {
        "age_band": pd.cut(
            frame["age_years"], bins=[0, 40, 50, 60, 120],
            labels=["under_40", "40_49", "50_59", "60_plus"],
            right=False,
        ).astype("string"),
        "gender": frame["gender"].map({1: "female", 2: "male"}).fillna("unknown").astype(str),
    }
    for dimension, labels in groups.items():
        report[dimension] = {}
        for label in sorted(labels.dropna().unique()):
            mask = labels.to_numpy() == label
            count = int(mask.sum())
            if count < 100:
                report[dimension][str(label)] = {"n": count, "suppressed": True}
                continue
            metrics = classification_metrics(np.asarray(y)[mask], np.asarray(p)[mask], threshold)
            report[dimension][str(label)] = {
                "n": count,
                "prevalence": float(np.mean(np.asarray(y)[mask])),
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "suppressed": False,
            }
    return report


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    raw_data = pd.read_csv(DATA, sep=";")
    data = clean_dataset(raw_data).reset_index(drop=True)
    x = data[FEATURE_COLUMNS].reset_index(drop=True)
    y = data["cardio"].astype(int).to_numpy()
    x_dev, _x_test, y_dev, _y_test = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y
    )
    x_dev = x_dev.reset_index(drop=True)
    dev_indices, _ = train_test_split(\n        np.arange(len(data)), test_size=0.20, random_state=SEED, stratify=y\n    )\n    dev_frame = data.iloc[dev_indices].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    model_factories = make_models()
    oof = {name: np.full(len(x_dev), np.nan) for name in model_factories}
    fold_rows = []

    for fold, (fit_idx, hold_idx) in enumerate(cv.split(x_dev, y_dev), start=1):
        x_fit = x_dev.iloc[fit_idx].reset_index(drop=True)
        x_hold = x_dev.iloc[hold_idx].reset_index(drop=True)
        y_fit, y_hold = y_dev[fit_idx], y_dev[hold_idx]
        config = fit_preprocessor(x_fit)
        fit_matrix = transform_frame(x_fit, config)
        hold_matrix = transform_frame(x_hold, config)
        for name, factory in model_factories.items():
            model = factory
            model.fit(fit_matrix, y_fit)
            probabilities = model.predict_proba(hold_matrix)[:, 1]
            oof[name][hold_idx] = probabilities
            metrics = classification_metrics(y_hold, probabilities)
            fold_rows.append({"fold": fold, "model": name, **metrics})
        print(f"Completed benchmark fold {fold}/{FOLDS}")

    summary = {}
    for name, probabilities in oof.items():
        if np.isnan(probabilities).any():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
        fold_metrics = [r for r in fold_rows if r["model"] == name]
        summary[name] = {
            "fold_mean_95ci": {
                metric: mean_ci([row[metric] for row in fold_metrics])
                for metric in ("roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "ece_10_bins")
            },
            "pooled_oof": classification_metrics(y_dev, probabilities),
            "roc_auc_bootstrap_95ci": bootstrap_auc_ci(y_dev, probabilities, "roc_auc"),
            "pr_auc_bootstrap_95ci": bootstrap_auc_ci(y_dev, probabilities, "pr_auc", seed=SEED + 1),
        }

    # Choose a benchmark reference using development-fold mean ROC-AUC only.
    selected = max(
        summary,
        key=lambda name: summary[name]["fold_mean_95ci"]["roc_auc"]["mean"],
    )
    selected_scores = oof[selected]
    selected_threshold = 0.5
    result = {
        "analysis_scope": "development partition only; final holdout deliberately not evaluated in this secondary benchmark",
        "dataset_rows_after_cleaning": int(len(data)),
        "data_quality": {
            "raw_rows": int(len(raw_data)),
            "rows_after_deterministic_cleaning": int(len(data)),
            "rows_removed_by_cleaning": int(len(raw_data) - len(data)),
            "duplicate_full_rows": int(raw_data.duplicated().sum()),
            "duplicate_ids": int(raw_data["id"].duplicated().sum()) if "id" in raw_data else None,
            "missing_values_by_column": {
                str(column): int(count)
                for column, count in raw_data.isna().sum().items()
            },
            "target_prevalence_after_cleaning": float(data["cardio"].mean()),
        },
        "development_rows": int(len(x_dev)),
        "folds": FOLDS,
        "selection_rule": "highest mean cross-validation ROC-AUC; descriptive benchmark only",
        "selected_benchmark_reference": selected,
        "models": summary,
        "fold_results": fold_rows,
        "threshold_analysis": threshold_table(y_dev, selected_scores),
        "subgroup_analysis": subgroup_report(dev_frame, y_dev, selected_scores, selected_threshold),
        "calibration": {
            "note": "Brier score, ECE and reliability bins describe probability reliability. Sigmoid-calibrated logistic regression is an explicit benchmark; no calibrated model replaces the deployed MLP.",
            "reliability_bins": expected_calibration_error(y_dev, selected_scores)[1],
            "selected_model": selected,
        },
        "limitations": [
            "The public dataset is cross-sectional and does not measure future cardiovascular events.",
            "All benchmark and subgroup estimates are internal cross-validation results, not external validation.",
            "Subgroup results are descriptive and are not proof of fairness or clinical equivalence.",
            "The original MLP test holdout has already been used by the existing project; this secondary benchmark intentionally does not re-use it.",
            "Confidence intervals quantify sampling variability under this split and do not account for dataset shift.",
        ],
    }
    (ARTIFACTS / "research_validation.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(fold_rows).to_csv(ARTIFACTS / "benchmark_folds.csv", index=False)
    pd.DataFrame(result["threshold_analysis"]).to_csv(
        ARTIFACTS / "threshold_analysis.csv", index=False
    )
    print(json.dumps({
        "selected_reference": selected,
        "models": {
            name: values["fold_mean_95ci"]["roc_auc"]
            for name, values in summary.items()
        },
        "artifacts": ["research_validation.json", "benchmark_folds.csv", "threshold_analysis.csv"],
    }, indent=2))


if __name__ == "__main__":
    main()
