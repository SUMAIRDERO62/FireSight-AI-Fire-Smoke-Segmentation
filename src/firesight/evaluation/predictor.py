"""Image preprocessing and prediction visualization utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(image_path, image_size, mean, std):
    """Load an RGB image and return normalized [1, 3, H, W] tensor."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError("image_size must contain positive height and width.")

    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean and std must contain three valid channels.")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    height, width = image_size
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    tensor = (tensor - mean_tensor) / std_tensor
    return tensor.unsqueeze(0), original_size


def create_color_mask(mask, num_classes):
    """Convert a 2D class-ID mask into an RGB color image."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array.")

    if np.any(mask < 0) or np.any(mask >= num_classes):
        raise ValueError("Mask contains class IDs outside the configured range.")

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        # Deterministic colors; class 0 is black/background.
        if class_id == 0:
            palette[class_id] = (0, 0, 0)
        else:
            palette[class_id] = (
                (class_id * 97) % 256,
                (class_id * 57 + 80) % 256,
                (class_id * 137 + 40) % 256,
            )

    return Image.fromarray(palette[mask.astype(np.int64)], mode="RGB")


def save_prediction(mask, output_path, num_classes, original_size=None):
    """Save class-ID mask as grayscale PNG."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Prediction mask must be two-dimensional.")

    if np.any(mask < 0) or np.any(mask >= num_classes):
        raise ValueError("Prediction contains invalid class IDs.")

    if num_classes > 256:
        raise ValueError("PNG grayscale masks support at most 256 class IDs.")

    image = Image.fromarray(mask.astype(np.uint8), mode="L")

    if original_size is not None:
        image = image.resize(original_size, Image.Resampling.NEAREST)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def save_overlay(
    original_image_path,
    mask,
    output_path,
    num_classes,
    alpha=0.45,
):
    """Blend a colored prediction mask over the original image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    original = Image.open(original_image_path).convert("RGB")
    color_mask = create_color_mask(mask, num_classes)
    color_mask = color_mask.resize(original.size, Image.Resampling.NEAREST)

    overlay = Image.blend(original, color_mask, alpha)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path
