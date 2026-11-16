"""Development-only operating-point frontier for the deployed MLP OOF scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RECALL_FLOORS = (0.95, 0.93, 0.90, 0.85, 0.80)

def operating_point_for_recall_floor(target, probabilities, recall_floor):
    """Choose the highest threshold whose OOF recall meets the requested floor."""
    y, scores = np.asarray(target, dtype=int), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("target and probabilities must be non-empty, aligned vectors")
    if not 0 < recall_floor <= 1:
        raise ValueError("recall_floor must be in (0, 1]")
    precision_curve, recall_curve, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(recall_curve[:-1] + 1e-12 >= recall_floor)
    if not len(eligible):
        raise RuntimeError(f"No threshold satisfies recall floor {recall_floor}")
    selected_index = int(eligible[-1])
    threshold = float(thresholds[selected_index])
    pred = (scores >= threshold).astype(int)
    recall = float(recall_score(y, pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    precision = float(precision_score(y, pred, zero_division=0))
    return {"recall_floor": float(recall_floor), "threshold": threshold,
        "recall": recall, "specificity": specificity, "precision": precision,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "scope": "development out-of-fold predictions only"}

def build_frontier(target, probabilities, recall_floors=RECALL_FLOORS):
    return [operating_point_for_recall_floor(target, probabilities, floor) for floor in recall_floors]

def main():
    source = pd.read_csv(ARTIFACTS / "mlp_oof_predictions.csv")
    rows = build_frontier(source["target"], source["oof_probability"])
    payload = {"method": "Highest OOF threshold satisfying each minimum recall floor",
        "recall_floors": list(RECALL_FLOORS), "rows": rows,
        "selection_scope": "development OOF predictions only",
        "warning": "Exploratory operating points; not evaluated on a fresh independent holdout and not clinical thresholds."}
    (ARTIFACTS / "mlp_operating_point_frontier.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(rows).to_csv(ARTIFACTS / "mlp_operating_point_frontier.csv", index=False)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
