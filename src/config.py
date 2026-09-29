"""Paths, seed, column lists, thresholds and env loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SEED = 42
RECALL_TARGET = 0.90
PRECISION_FLOOR = 0.70
N_SPLITS = 5
N_ITER = 50
N_BOOT = 1000

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://heart:heart@localhost:5433/heart")
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "false").lower() == "true"

RAW_FILE = ROOT / "data" / "raw" / "processed.cleveland.data"
RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
NOMINAL = ["cp", "restecg", "thal", "slope"]
BINARY = ["sex", "fbs", "exang"]

VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}
NUM_CODES = {0, 1, 2, 3, 4}
EXPECTED_ROWS = 303
EXPECTED_MISSING = {"ca": 4, "thal": 2}
