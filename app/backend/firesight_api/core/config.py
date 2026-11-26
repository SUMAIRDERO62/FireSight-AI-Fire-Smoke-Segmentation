"""Central application configuration for FireSight AI."""

from pathlib import Path


# Repository root:
# app/backend/firesight_api/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]

BACKEND_ROOT = PROJECT_ROOT / "app" / "backend"

# Application
APP_NAME = "FireSight AI"
APP_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# Model checkpoints
UNET_CHECKPOINT = (
    PROJECT_ROOT / "models" / "checkpoints" / "phase05_unet" / "unet_best.pth"
)

YOLO_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "firesight_phase05_fair_yolo11n_seg"
    / "weights"
    / "best.pt"
)

# U-Net inference settings
UNET_INPUT_SIZE = (256, 256)
UNET_NUM_CLASSES = 2
UNET_INPUT_CHANNELS = 3
UNET_BASE_CHANNELS = 16
UNET_THRESHOLD = 0.5

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# YOLO11-seg inference settings
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25

# Upload validation
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
}

# Storage
RESULTS_DIR = PROJECT_ROOT / "outputs" / "api_results"

# Device
DEVICE = "cpu"
