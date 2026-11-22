import numpy as np
import pytest
import torch
from PIL import Image

from firesight.evaluation.predictor import (
    load_image,
    create_color_mask,
    save_prediction,
    save_overlay,
)


def test_load_image_returns_normalized_batch(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), color=(255, 0, 0)).save(path)

    tensor, original_size = load_image(
        path,
        image_size=(16, 16),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    assert tensor.shape == (1, 3, 16, 16)
    assert original_size == (32, 24)
    assert torch.isfinite(tensor).all()


def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(
            tmp_path / "missing.jpg",
            (16, 16),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )


def test_color_mask_returns_rgb_image():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    result = create_color_mask(mask, num_classes=2)

    assert result.mode == "RGB"
    assert result.size == (2, 2)


def test_save_prediction_creates_grayscale_png(tmp_path):
    mask = torch.tensor([[0, 1], [1, 0]])
    output = tmp_path / "predictions" / "mask.png"

    saved_path = save_prediction(mask, output, num_classes=2, original_size=(8, 8))

    assert saved_path.is_file()
    with Image.open(saved_path) as result:
        assert result.mode == "L"
        assert result.size == (8, 8)


def test_save_overlay_creates_image(tmp_path):
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (20, 20), color=(100, 100, 100)).save(image_path)

    mask = np.ones((10, 10), dtype=np.uint8)
    output = tmp_path / "overlay.png"

    save_overlay(image_path, mask, output, num_classes=2)

    assert output.is_file()
    with Image.open(output) as result:
        assert result.mode == "RGB"
        assert result.size == (20, 20)


def test_invalid_mask_class_is_rejected():
    with pytest.raises(ValueError, match="class IDs"):
        create_color_mask(
            np.array([[0, 3]]),
            num_classes=2,
        )


def test_invalid_overlay_alpha_is_rejected(tmp_path):
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (10, 10)).save(image_path)

    with pytest.raises(ValueError, match="alpha"):
        save_overlay(
            image_path,
            np.zeros((10, 10), dtype=np.uint8),
            tmp_path / "overlay.png",
            num_classes=2,
            alpha=2.0,
        )
