from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Dataset for genuine pixel-level semantic segmentation masks."""

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        num_classes: Optional[int] = None,
        image_size: Optional[tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if image_size is not None:
            if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
                raise ValueError("image_size must be positive (height, width).")

        self.samples = self._discover_samples()

        if not self.samples:
            raise ValueError("No image-mask pairs were found.")

    def _discover_samples(self):
        images = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
        )

        samples = []
        missing = []

        for image_path in images:
            mask_path = self.masks_dir / f"{image_path.stem}.png"

            if not mask_path.is_file():
                missing.append(image_path.name)
            else:
                samples.append((image_path, mask_path))

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have missing PNG masks. Examples: {missing[:5]}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as file:
            image = file.convert("RGB")

        with Image.open(mask_path) as file:
            mask = file.copy()

        if mask.mode not in ("L", "I", "I;16"):
            raise ValueError(
                f"Mask must be single-channel integer data: "
                f"{mask_path}, mode={mask.mode}"
            )

        if image.size != mask.size:
            raise ValueError(f"Image-mask size mismatch: {image_path.name}")

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask)

        if mask_array.ndim != 2:
            raise ValueError("Mask must be a two-dimensional array.")

        if not np.issubdtype(mask_array.dtype, np.integer):
            raise ValueError("Mask values must be integers.")

        if np.any(mask_array < 0):
            raise ValueError("Negative mask class IDs are not allowed.")

        if self.num_classes is not None:
            if np.any(mask_array >= self.num_classes):
                raise ValueError(
                    f"Invalid class ID in {mask_path}. "
                    f"Expected IDs from 0 to {self.num_classes - 1}."
                )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_array.transpose(2, 0, 1))
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_array.astype(np.int64))
        )

        if self.image_size is not None:
            target_h, target_w = self.image_size

            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_tensor = (
                torch.nn.functional.interpolate(
                    mask_tensor[None, None].float(),
                    size=(target_h, target_w),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
