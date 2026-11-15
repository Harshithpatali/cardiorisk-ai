"""Persist the final model, preprocessing and evaluation outputs."""
import json
from pathlib import Path
import pandas as pd
import torch


def save_artifacts(model, preprocessing, report, history, output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "mlp_state_dict.pt")
    (destination / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2)
    )
    (destination / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(history).to_csv(
        destination / "training_history.csv",
        index=False,
    )
    print(f"Artifacts saved to {destination}")
