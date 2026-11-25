"""Two-channel fire/smoke dataset preserving class overlap."""
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class MultiLabelSegmentationDataset(Dataset):
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        images_dir: str | Path,
        fire_masks_dir: str | Path,
        smoke_masks_dir: str | Path,
        image_size: Optional[tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.fire_masks_dir = Path(fire_masks_dir)
        self.smoke_masks_dir = Path(smoke_masks_dir)
        self.image_size = image_size
        self.transform = transform
        for directory in (self.images_dir, self.fire_masks_dir, self.smoke_masks_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Directory not found: {directory}")
        self.samples = []
        images = sorted(p for p in self.images_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS)
        missing = []
        for image in images:
            fire = self.fire_masks_dir / f"{image.stem}.png"
            smoke = self.smoke_masks_dir / f"{image.stem}.png"
            if not fire.is_file() or not smoke.is_file():
                missing.append(image.name)
            else:
                self.samples.append((image, fire, smoke))
        if missing:
            raise FileNotFoundError(f"Missing fire/smoke masks for: {missing[:5]}")
        if not self.samples:
            raise ValueError("No image and two-mask samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, fire_path, smoke_path = self.samples[index]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(fire_path) as im:
            fire = im.convert("L")
        with Image.open(smoke_path) as im:
            smoke = im.convert("L")
        if image.size != fire.size or image.size != smoke.size:
            raise ValueError(f"Image/mask size mismatch: {image_path.name}")

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        masks_np = np.stack(
            (np.asarray(fire, dtype=np.uint8), np.asarray(smoke, dtype=np.uint8)),
            axis=0,
        )
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(masks_np > 0)).float()

        if self.image_size:
            image_tensor = F.interpolate(
                image_tensor[None], size=self.image_size, mode="bilinear",
                align_corners=False,
            )[0]
            mask_tensor = F.interpolate(
                mask_tensor[None], size=self.image_size, mode="nearest",
            )[0]

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return image_tensor, mask_tensor
