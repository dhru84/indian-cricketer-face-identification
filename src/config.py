from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # data/raw/<player_name>/*.jpg
CACHE_DIR = ROOT / "data" / "cache"
MODEL_PATH = ROOT / "models" / "face_classifier.joblib"
RESULTS_DIR = ROOT / "results"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
DEDUP_HAMMING = 6        # perceptual-hash distance <= this => near-duplicate
TEST_SIZE = 0.25
SEED = 42
UNKNOWN_THRESHOLD = 0.55  # default; train.py tunes and saves a better one
UNKNOWN_LABEL = "Unknown"
