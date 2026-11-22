import torch

from firesight.training.losses import (
    DiceLoss,
    CrossEntropyDiceLoss,
)


def test_dice_loss_returns_valid_value():
    logits = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 3, (2, 32, 32))

    loss = DiceLoss()(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0 <= loss.item() <= 1


def test_combined_loss_returns_valid_value():
    logits = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 3, (2, 32, 32))

    criterion = CrossEntropyDiceLoss()
    loss = criterion(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_combined_loss_supports_backpropagation():
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))

    loss = CrossEntropyDiceLoss()(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_loss_supports_ignore_index():
    logits = torch.randn(1, 3, 16, 16)
    targets = torch.randint(0, 3, (1, 16, 16))

    targets[:, 0, :] = 255

    loss = DiceLoss(ignore_index=255)(logits, targets)

    assert torch.isfinite(loss)
