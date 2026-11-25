import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Legacy multiclass soft Dice loss."""
    def __init__(self, ignore_index=None, smooth=1e-6):
        super().__init__()
        self.ignore_index, self.smooth = ignore_index, smooth

    def forward(self, logits, targets):
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError("Expected logits [N,C,H,W] and targets [N,H,W].")
        if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError("Batch or spatial dimensions do not match.")
        classes = logits.shape[1]
        valid = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
        safe = targets.masked_fill(~valid, 0)
        if torch.any(safe[valid] < 0) or torch.any(safe[valid] >= classes):
            raise ValueError("Target class ID is outside logits range.")
        probs = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
        onehot = F.one_hot(safe.long(), classes).permute(0, 3, 1, 2).float()
        onehot = onehot * valid.unsqueeze(1)
        probs, onehot = probs.flatten(2), onehot.flatten(2)
        intersection = (probs * onehot).sum(2)
        denominator = probs.sum(2) + onehot.sum(2)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    """Legacy combined multiclass cross-entropy and Dice loss."""
    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=None):
        super().__init__()
        if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight == 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")
        self.ce_weight, self.dice_weight = ce_weight, dice_weight
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce_weight * self.ce_loss(logits, targets) + \
            self.dice_weight * self.dice_loss(logits, targets)


class MultiLabelDiceLoss(nn.Module):
    """Independent sigmoid Dice loss; overlapping foreground classes are allowed."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.ndim != 4 or targets.shape != logits.shape:
            raise ValueError("Expected matching [N,C,H,W] logits and targets.")
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        denominator = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class BCEWithLogitsDiceLoss(nn.Module):
    """BCE-with-logits plus independent per-class soft Dice."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiLabelDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError("Multi-label targets must match logits shape.")
        targets = targets.to(dtype=logits.dtype)
        return self.bce_weight * self.bce(logits, targets) + \
            self.dice_weight * self.dice(logits, targets)
