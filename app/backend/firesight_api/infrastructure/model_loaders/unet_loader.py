"""Load the trained Phase 05 U-Net segmentation model."""

import sys
from pathlib import Path

import torch

from ...core.config import (
    DEVICE,
    UNET_BASE_CHANNELS,
    UNET_CHECKPOINT,
    UNET_INPUT_CHANNELS,
    UNET_NUM_CLASSES,
)


def load_unet_model() -> torch.nn.Module:
    """Build the trained U-Net architecture and load its checkpoint."""

    checkpoint_path = Path(UNET_CHECKPOINT)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {checkpoint_path}")

    # Make the project's src directory available for model imports.
    source_path = str(checkpoint_path.parents[4] / "src")

    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    from firesight.models.unet import UNet

    device = torch.device(DEVICE)

    model = UNet(
        num_classes=UNET_NUM_CLASSES,
        in_channels=UNET_INPUT_CHANNELS,
        base_channels=UNET_BASE_CHANNELS,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid U-Net checkpoint format.")

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("U-Net checkpoint does not contain model_state_dict.")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model
