from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_TARGETS = 206
