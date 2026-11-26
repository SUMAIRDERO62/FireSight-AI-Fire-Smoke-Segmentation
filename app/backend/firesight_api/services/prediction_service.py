
"""Run FireSight AI segmentation inference."""

import time

import numpy as np
import torch
from PIL import Image

from ..core.config import (
    DEVICE,
    UNET_THRESHOLD,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
)
from ..models.model_registry import get_model
from .image_service import prepare_unet_input, prepare_yolo_input
from .mask_service import process_unet_output, process_yolo_output
from .overlay_service import create_overlay


SUPPORTED_MODELS = ("unet", "yolo11-seg")


def _resize_masks(
    masks: dict[str, np.ndarray],
    output_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Resize binary masks to the original image dimensions."""

    resized = {}

    for class_name, mask in masks.items():
        mask_image = Image.fromarray(
            (mask.astype(np.uint8) * 255),
            mode="L",
        )

        mask_image = mask_image.resize(
            output_size,
            Image.Resampling.NEAREST,
        )

        resized[class_name] = (
            np.asarray(mask_image) > 0
        ).astype(np.uint8)

    return resized


def predict_image(
    image: Image.Image,
    model_name: str,
) -> dict:
    """Run segmentation inference and return masks, overlay and metadata."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose one of: {', '.join(SUPPORTED_MODELS)}."
        )

    original = image.convert("RGB")
    original_width, original_height = original.size

    start_time = time.perf_counter()

    model = get_model(model_name)

    if model_name == "unet":
        input_tensor = prepare_unet_input(original)

        with torch.inference_mode():
            logits = model(input_tensor.to(DEVICE))

        model_masks = process_unet_output(
            logits,
            threshold=UNET_THRESHOLD,
        )

    else:
        yolo_image = prepare_yolo_input(original)

        results = model.predict(
            source=yolo_image,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        model_masks = process_yolo_output(
            results,
            output_size=(original_height, original_width),
        )

    masks = _resize_masks(
        model_masks,
        output_size=(original_width, original_height),
    )

    overlay = create_overlay(original, masks)

    elapsed_seconds = time.perf_counter() - start_time

    class_results = {}

    for class_name, mask in masks.items():
        class_results[class_name] = {
            "detected": bool(mask.any()),
            "pixel_count": int(mask.sum()),
            "coverage_percent": round(
                float(mask.mean() * 100),
                4,
            ),
        }

    return {
        "model": model_name,
        "image_width": original_width,
        "image_height": original_height,
        "processing_time_seconds": round(elapsed_seconds, 6),
        "classes": class_results,
        "masks": masks,
        "overlay": overlay,
    }
