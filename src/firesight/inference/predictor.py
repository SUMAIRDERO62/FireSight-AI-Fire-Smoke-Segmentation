"""Shared image segmentation predictor."""

import logging
from pathlib import Path

import numpy as np

from firesight.models.segmentation import (
    ModelNotAvailableError,
    SegmentationModel,
    SegmentationResult,
    create_segmentation_model,
)
from firesight.preprocessing.image import load_image

LOGGER = logging.getLogger(__name__)


class Predictor:
    """Load and invoke one supported image segmentation model."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str | Path,
        image_size: int = 512,
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        LOGGER.info("Initializing %s segmentation predictor", model_name)
        self.model: SegmentationModel = create_segmentation_model(
            model_name, checkpoint
        )

    def predict_array(self, image: np.ndarray) -> SegmentationResult:
        return self.model.predict(image)

    def predict_bytes(self, data: bytes) -> SegmentationResult:
        image = load_image(data, image_size=None)
        return self.predict_array(image)

    def predict_file(self, path: str | Path) -> SegmentationResult:
        return self.predict_bytes(Path(path).read_bytes())

