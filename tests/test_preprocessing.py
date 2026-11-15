import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from preprocessing import clean_dataset, make_feature_row, transform_frame


def test_cleaning_and_target_shape():
    data = clean_dataset(ROOT / "data" / "cardio_train.csv")
    assert len(data) > 1000
    assert set(data.cardio.unique()) == {0, 1}


def test_input_transform_matches_model_dimension():
    cfg = json.loads((ROOT / "artifacts" / "preprocessing.json").read_text())
    row = make_feature_row(55, 1, 165, 75, 120, 80, 1, 1, 0, 0, 1)
    x = transform_frame(row, cfg)
    assert x.shape == (1, cfg["input_dim"]) and np.isfinite(x).all()


def test_model_artifact_loads_and_predicts():
    import torch
    from model import CardioMLP

    cfg = json.loads((ROOT / "artifacts" / "preprocessing.json").read_text())
    state = torch.load(ROOT / "artifacts" / "mlp_state_dict.pt", map_location="cpu", weights_only=True)
    net = CardioMLP(cfg["input_dim"])
    net.load_state_dict(state)
    net.eval()
    x = transform_frame(make_feature_row(55, 1, 165, 75, 120, 80, 1, 1, 0, 0, 1), cfg)
    with torch.no_grad():
        score = torch.sigmoid(net(torch.tensor(x, dtype=torch.float32))).item()
    assert 0 <= score <= 1
