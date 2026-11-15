"""Shared configuration for the cardio MLP pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cardio_train.csv"
ARTIFACT_DIR = ROOT / "artifacts"
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
MAX_EPOCHS = 45
EARLY_STOPPING_PATIENCE = 7
BATCH_SIZE = 512
MIN_OOF_PRECISION = 0.55
