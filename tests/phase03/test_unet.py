import pytest
import torch

from firesight.models.unet import UNet


@pytest.fixture
def model():
    """Create a lightweight U-Net for CPU testing."""
    return UNet(
        num_classes=3,
        in_channels=3,
        base_channels=8,
    )


def test_unet_output_shape(model):
    """Output must preserve the input spatial dimensions."""
    images = torch.randn(1, 3, 128, 160)

    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (1, 3, 128, 160)


def test_unet_supports_odd_dimensions(model):
    """U-Net must handle images with odd height and width."""
    images = torch.randn(1, 3, 127, 135)

    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (1, 3, 127, 135)


def test_unet_supports_multiple_images(model):
    """Model must process a batch of images."""
    images = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (2, 3, 128, 128)


def test_unet_has_trainable_parameters(model):
    """Model must contain trainable parameters."""
    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert sum(p.numel() for p in parameters) > 0
    assert all(p.requires_grad for p in parameters)


def test_unet_supports_backpropagation(model):
    """Gradients must flow through the complete network."""
    model.train()

    images = torch.randn(1, 3, 64, 64)
    outputs = model(images)
    loss = outputs.square().mean()

    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]

    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )
