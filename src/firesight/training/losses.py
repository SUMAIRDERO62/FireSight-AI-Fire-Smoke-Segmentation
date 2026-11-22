import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multiclass soft Dice loss."""

    def __init__(self, ignore_index=None, smooth=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.ndim != 4:
            raise ValueError("Logits must have shape [N, C, H, W].")

        if targets.ndim != 3:
            raise ValueError("Targets must have shape [N, H, W].")

        if logits.shape[0] != targets.shape[0]:
            raise ValueError("Batch sizes do not match.")

        if logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError("Spatial dimensions do not match.")

        num_classes = logits.shape[1]

        valid = torch.ones_like(targets, dtype=torch.bool)

        if self.ignore_index is not None:
            valid = targets != self.ignore_index

        safe_targets = targets.masked_fill(~valid, 0)

        if torch.any(safe_targets[valid] < 0) or torch.any(
            safe_targets[valid] >= num_classes
        ):
            raise ValueError("Target class ID is outside logits range.")

        probabilities = torch.softmax(logits, dim=1)

        one_hot = (
            F.one_hot(
                safe_targets.long(),
                num_classes=num_classes,
            )
            .permute(0, 3, 1, 2)
            .float()
        )

        valid = valid.unsqueeze(1)

        probabilities = probabilities * valid
        one_hot = one_hot * valid

        probabilities = probabilities.flatten(2)
        one_hot = one_hot.flatten(2)

        intersection = (probabilities * one_hot).sum(dim=2)

        denominator = probabilities.sum(dim=2) + one_hot.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    """Combined Cross Entropy and Dice loss."""

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        ignore_index=None,
    ):
        super().__init__()

        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights cannot be negative.")

        if ce_weight + dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        if ignore_index is None:
            self.ce_loss = nn.CrossEntropyLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = self.ce_weight * ce + self.dice_weight * dice

        return total
