
"""Image preprocessing for FireSight AI segmentation models."""

import numpy as np
import torch
from PIL import Image

from ..core.config import (
    IMAGE_MEAN,
    IMAGE_STD,
    UNET_INPUT_SIZE,
)


def prepare_unet_input(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image into a normalized U-Net input tensor."""

    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = image.resize(
        UNET_INPUT_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(resized, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


def prepare_yolo_input(image: Image.Image) -> Image.Image:
    """Return an RGB image for Ultralytics YOLO inference."""

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image.copy()


def get_image_metadata(image: Image.Image) -> dict:
    """Return basic metadata about the uploaded image."""

    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
    }
