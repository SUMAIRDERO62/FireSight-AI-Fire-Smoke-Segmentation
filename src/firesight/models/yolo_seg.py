"""YOLO11 instance-segmentation model factory."""

from pathlib import Path


def build_yolo_seg_model(weights: str = "yolo11n-seg.pt"):
    """Load a pretrained Ultralytics segmentation model."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install it with: "
            "python -m pip install ultralytics"
        ) from exc

    weights_path = Path(weights)
    if weights_path.is_dir():
        raise ValueError(f"Model weights must be a file or model name: {weights}")

    return YOLO(str(weights), task="segment")
