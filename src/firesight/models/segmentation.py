"""Segmentation model interfaces for U-Net and YOLO11-seg."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested checkpoint or implementation is unavailable."""


class SegmentationDependencyError(RuntimeError):
    """Raised when an optional segmentation dependency is missing."""


@dataclass(frozen=True)
class SegmentationResult:
    """One image segmentation result."""

    mask: np.ndarray
    labels: tuple[str, ...]
    scores: tuple[float, ...]


class SegmentationModel(Protocol):
    """Shared contract implemented by every segmentation adapter."""

    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Segment a single RGB image."""


class UnetSegmentationModel:
    """U-Net checkpoint adapter."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise ModelNotAvailableError(
                f"U-Net checkpoint is unavailable: {self.checkpoint}"
            )

    def predict(self, image: np.ndarray) -> SegmentationResult:
        raise NotImplementedError(
            "U-Net checkpoint loading and architecture inference are not implemented."
        )


class Yolo11SegmentationModel:
    """YOLO11-seg checkpoint adapter."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise ModelNotAvailableError(
                f"YOLO11-seg checkpoint is unavailable: {self.checkpoint}"
            )

    def predict(self, image: np.ndarray) -> SegmentationResult:
        raise NotImplementedError(
            "YOLO11-seg checkpoint loading and inference are not implemented."
        )


def create_segmentation_model(
    model_name: str, checkpoint: str | Path
) -> SegmentationModel:
    """Create one supported segmentation model adapter."""
    normalized = model_name.strip().lower()
    if normalized == "unet":
        return UnetSegmentationModel(checkpoint)
    if normalized == "yolo11-seg":
        return Yolo11SegmentationModel(checkpoint)
    raise ValueError(f"Unsupported segmentation model: {model_name}")
