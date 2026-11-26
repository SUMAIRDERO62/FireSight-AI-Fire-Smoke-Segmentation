
"""Load the trained Phase 05 YOLO11-seg model."""

from pathlib import Path

from ultralytics import YOLO

from ...core.config import YOLO_CHECKPOINT


def load_yolo_model() -> YOLO:
    """Load the trained YOLO11-seg checkpoint."""

    checkpoint_path = Path(YOLO_CHECKPOINT)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"YOLO11-seg checkpoint not found: {checkpoint_path}"
        )

    model = YOLO(str(checkpoint_path), task="segment")

    if model.task != "segment":
        raise RuntimeError(
            f"Expected segmentation model, received: {model.task}"
        )

    return model
