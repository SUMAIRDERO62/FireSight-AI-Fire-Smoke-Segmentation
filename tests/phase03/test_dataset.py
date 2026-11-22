import numpy as np
import pytest
import torch
from PIL import Image

from firesight.data.segmentation_dataset import SegmentationDataset
from firesight.data.transforms import SegmentationTransform


def create_sample(root, name="sample"):
    images = root / "images"
    masks = root / "masks"

    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    image = np.zeros((64, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 120

    mask = np.zeros((64, 80), dtype=np.uint8)
    mask[20:40, 20:50] = 1

    Image.fromarray(image).save(images / f"{name}.jpg")
    Image.fromarray(mask).save(masks / f"{name}.png")

    return images, masks


def test_dataset_loads_image_and_mask(tmp_path):
    images, masks = create_sample(tmp_path)

    dataset = SegmentationDataset(images, masks, num_classes=2)

    image, mask = dataset[0]

    assert image.shape == (3, 64, 80)
    assert mask.shape == (64, 80)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert set(torch.unique(mask).tolist()) == {0, 1}


def test_dataset_resizes_image_and_mask(tmp_path):
    images, masks = create_sample(tmp_path)

    dataset = SegmentationDataset(
        images,
        masks,
        num_classes=2,
        image_size=(128, 128),
    )

    image, mask = dataset[0]

    assert image.shape == (3, 128, 128)
    assert mask.shape == (128, 128)
    assert set(torch.unique(mask).tolist()) <= {0, 1}


def test_dataset_rejects_missing_mask(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"

    images.mkdir()
    masks.mkdir()

    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images / "missing.jpg")

    with pytest.raises(FileNotFoundError):
        SegmentationDataset(images, masks)


def test_dataset_rejects_invalid_class_id(tmp_path):
    images, masks = create_sample(tmp_path)

    invalid_mask = np.full((64, 80), 3, dtype=np.uint8)
    Image.fromarray(invalid_mask).save(masks / "sample.png")

    dataset = SegmentationDataset(images, masks, num_classes=2)

    with pytest.raises(ValueError, match="class ID"):
        dataset[0]


def test_transform_preserves_mask_class_ids(tmp_path):
    images, masks = create_sample(tmp_path)

    transform = SegmentationTransform(
        image_size=(96, 96),
        horizontal_flip_prob=0,
        vertical_flip_prob=0,
    )

    dataset = SegmentationDataset(
        images,
        masks,
        num_classes=2,
        transform=transform,
    )

    image, mask = dataset[0]

    assert image.shape == (3, 96, 96)
    assert mask.shape == (96, 96)
    assert mask.dtype == torch.int64
    assert set(torch.unique(mask).tolist()) <= {0, 1}
