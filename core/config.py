from pathlib import Path
import logging

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

log = logging.getLogger("pharmacheck")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_ROOT = Path(
    "/Users/mac/Desktop/capstone-prep/Counterfeit_med_detection.v1i.multiclass"
)

ARTIFACT_DIR = Path(
    "/Users/mac/Desktop/capstone-prep/artifacts"
    
)

HISTORY_PATH = ARTIFACT_DIR / "prediction_history.jsonl"

MODEL_KERAS = ARTIFACT_DIR / "pharmacheck_mobilenetv3.keras"
MODEL_TFLITE = Path("artifacts/pharmacheck_mobilenetv3.tflite")
# ------------------------------------------------------------------
# Model settings
# ------------------------------------------------------------------

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "authentic",
    "counterfeit",
]

SPLITS = [
    "train",
    "valid",
    "test",
]

CONF_THRESHOLD = 0.50

# ------------------------------------------------------------------
# Upload settings
# ------------------------------------------------------------------

MAX_BATCH_IMAGES = 20

MAX_IMAGE_BYTES = 10 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

# ------------------------------------------------------------------
# Visualisation colours
# ------------------------------------------------------------------

TERRACOTTA = "#C0614C"
SUCCESS = "#38A169"
SLATE = "#4A5568"