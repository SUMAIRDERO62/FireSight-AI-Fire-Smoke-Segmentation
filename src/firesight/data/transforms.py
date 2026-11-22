import random
from typing import Optional

import torch
import torch.nn.functional as F


class SegmentationTransform:
    """Apply paired preprocessing to image and segmentation mask."""

    def __init__(
        self,
        image_size: Optional[tuple[int, int]] = None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
    ):
        if not 0 <= horizontal_flip_prob <= 1:
            raise ValueError("Invalid horizontal flip probability.")

        if not 0 <= vertical_flip_prob <= 1:
            raise ValueError("Invalid vertical flip probability.")

        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Mean and std must contain 3 channels.")

        if any(value <= 0 for value in std):
            raise ValueError("Standard deviations must be positive.")

        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Image must have shape [3, H, W].")

        if mask.ndim != 2:
            raise ValueError("Mask must have shape [H, W].")

        if image.shape[-2:] != mask.shape:
            raise ValueError("Image and mask dimensions must match.")

        image = image.float()
        mask = mask.long()

        if image.min() < 0 or image.max() > 1:
            raise ValueError("Image values must be in the [0, 1] range.")

        if self.image_size is not None:
            height, width = self.image_size

            image = F.interpolate(
                image.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask = (
                F.interpolate(
                    mask[None, None].float(),
                    size=(height, width),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        if random.random() < self.horizontal_flip_prob:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if random.random() < self.vertical_flip_prob:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])

        image = (image - self.mean.to(image.device)) / self.std.to(image.device)

        return image.contiguous(), mask.contiguous()
