
"""Generate visual segmentation overlays for FireSight AI."""

from PIL import Image

import numpy as np


FIRE_COLOR = (255, 40, 40)
SMOKE_COLOR = (40, 120, 255)
OVERLAY_ALPHA = 0.45


def create_overlay(
    image: Image.Image,
    masks: dict[str, np.ndarray],
    alpha: float = OVERLAY_ALPHA,
) -> Image.Image:
    """Blend fire and smoke segmentation masks over an RGB image."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")

    original = image.convert("RGB")
    original_array = np.asarray(original, dtype=np.float32).copy()

    height, width = original_array.shape[:2]

    fire_mask = masks.get("fire")
    smoke_mask = masks.get("smoke")

    overlay = np.zeros_like(original_array)

    if fire_mask is not None:
        if fire_mask.shape != (height, width):
            raise ValueError("Fire mask dimensions do not match the image.")

        overlay[fire_mask.astype(bool)] = FIRE_COLOR

    if smoke_mask is not None:
        if smoke_mask.shape != (height, width):
            raise ValueError("Smoke mask dimensions do not match the image.")

        smoke_pixels = smoke_mask.astype(bool)

        # Average colors in overlapping fire and smoke regions.
        fire_pixels = (
            fire_mask.astype(bool)
            if fire_mask is not None
            else np.zeros((height, width), dtype=bool)
        )

        overlap = fire_pixels & smoke_pixels
        overlay[smoke_pixels & ~overlap] = SMOKE_COLOR
        overlay[overlap] = (
            np.array(FIRE_COLOR, dtype=np.float32)
            + np.array(SMOKE_COLOR, dtype=np.float32)
        ) / 2

    active_mask = np.zeros((height, width), dtype=bool)

    if fire_mask is not None:
        active_mask |= fire_mask.astype(bool)

    if smoke_mask is not None:
        active_mask |= smoke_mask.astype(bool)

    result = original_array.copy()

    result[active_mask] = (
        original_array[active_mask] * (1.0 - alpha)
        + overlay[active_mask] * alpha
    )

    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result, mode="RGB")
