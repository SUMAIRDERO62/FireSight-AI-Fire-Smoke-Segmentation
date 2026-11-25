import random
from typing import Optional

import torch
import torch.nn.functional as F


class SegmentationTransform:
    """Paired preprocessing for legacy integer masks and multi-label masks."""

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
        if image_size is not None and (len(image_size) != 2 or min(image_size) <= 0):
            raise ValueError("image_size must be positive (height, width).")
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Image must have shape [3, H, W].")
        if mask.ndim not in (2, 3):
            raise ValueError("Mask must have shape [H,W] or [C,H,W].")
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("Image and mask dimensions must match.")
        image = image.float()
        multilabel = mask.ndim == 3
        mask = mask.float() if multilabel else mask.long()
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Image values must be in the [0, 1] range.")
        if multilabel and (mask.min() < 0 or mask.max() > 1):
            raise ValueError("Multi-label mask values must be in [0, 1].")

        if self.image_size is not None:
            image = F.interpolate(
                image[None], size=self.image_size, mode="bilinear",
                align_corners=False,
            )[0]
            if multilabel:
                mask = F.interpolate(mask[None], size=self.image_size, mode="nearest")[0]
            else:
                mask = F.interpolate(
                    mask[None, None].float(), size=self.image_size, mode="nearest"
                )[0, 0].long()

        if random.random() < self.horizontal_flip_prob:
            image, mask = torch.flip(image, [2]), torch.flip(mask, [-1])
        if random.random() < self.vertical_flip_prob:
            image, mask = torch.flip(image, [1]), torch.flip(mask, [-2])

        image = (image - self.mean.to(image.device)) / self.std.to(image.device)
        return image.contiguous(), mask.contiguous()
