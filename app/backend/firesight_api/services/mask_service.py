
"""Convert model outputs into fire and smoke segmentation masks."""

import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import UNET_THRESHOLD


CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}


def process_unet_output(
    logits: torch.Tensor,
    threshold: float = UNET_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Convert U-Net logits into independent binary class masks."""

    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(
            "Expected U-Net output with shape (1, 2, height, width)."
        )

    if logits.shape[1] != len(CLASS_NAMES):
        raise ValueError("U-Net output must contain fire and smoke channels.")

    probabilities = torch.sigmoid(logits)[0]

    masks = {}

    for class_id, class_name in CLASS_NAMES.items():
        masks[class_name] = (
            probabilities[class_id] >= threshold
        ).cpu().numpy().astype(np.uint8)

    return masks


def process_yolo_output(
    results,
    output_size: tuple[int, int] = (256, 256),
) -> dict[str, np.ndarray]:
    """Combine YOLO instance masks into independent class masks."""

    height, width = output_size

    masks = {
        "fire": np.zeros((height, width), dtype=np.uint8),
        "smoke": np.zeros((height, width), dtype=np.uint8),
    }

    if not results:
        return masks

    result = results[0]

    if result.masks is None or result.boxes is None:
        return masks

    instance_masks = result.masks.data

    resized_masks = F.interpolate(
        instance_masks.unsqueeze(1).float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    class_ids = result.boxes.cls.int().cpu().tolist()

    for index, class_id in enumerate(class_ids):
        class_name = CLASS_NAMES.get(class_id)

        if class_name is None:
            continue

        binary_mask = (
            resized_masks[index].cpu().numpy() > 0.5
        ).astype(np.uint8)

        masks[class_name] = np.maximum(
            masks[class_name],
            binary_mask,
        )

    return masks
