"""Evaluation metrics for semantic segmentation."""

import torch


class SegmentationMetrics:
    """Calculate confusion matrix, Dice, IoU, precision and recall."""

    def __init__(self, num_classes: int, ignore_index=None):
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, predictions, targets):
        """Update confusion matrix using logits or predicted class IDs."""

        if not isinstance(predictions, torch.Tensor):
            raise TypeError("predictions must be a torch.Tensor.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("targets must be a torch.Tensor.")

        if predictions.ndim == targets.ndim + 1:
            if predictions.shape[1] != self.num_classes:
                raise ValueError("Logit channel count does not match num_classes.")
            predictions = predictions.argmax(dim=1)
        elif predictions.ndim != targets.ndim:
            raise ValueError("Predictions and targets have incompatible dimensions.")

        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes must match.")

        predictions = predictions.to(torch.long).reshape(-1).cpu()
        targets = targets.to(torch.long).reshape(-1).cpu()

        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            predictions = predictions[valid]
            targets = targets[valid]

        if targets.numel() == 0:
            return

        if torch.any((targets < 0) | (targets >= self.num_classes)):
            raise ValueError("Targets contain class IDs outside the configured range.")

        if torch.any((predictions < 0) | (predictions >= self.num_classes)):
            raise ValueError(
                "Predictions contain class IDs outside the configured range."
            )

        encoded = targets * self.num_classes + predictions
        batch_matrix = torch.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

        self.confusion_matrix += batch_matrix

    def compute(self):
        """Return global and per-class segmentation metrics."""

        matrix = self.confusion_matrix.to(torch.float64)
        tp = torch.diag(matrix)
        actual = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        fp = predicted - tp
        fn = actual - tp

        def divide(numerator, denominator):
            values = []
            for n, d in zip(numerator.tolist(), denominator.tolist()):
                values.append(n / d if d > 0 else None)
            return values

        precision = divide(tp, predicted)
        recall = divide(tp, actual)
        dice = divide(2 * tp, 2 * tp + fp + fn)
        iou = divide(tp, tp + fp + fn)

        def macro(values):
            valid = [value for value in values if value is not None]
            return sum(valid) / len(valid) if valid else None

        per_class = []
        for class_id in range(self.num_classes):
            per_class.append(
                {
                    "class_id": class_id,
                    "support": int(actual[class_id].item()),
                    "predicted_pixels": int(predicted[class_id].item()),
                    "precision": precision[class_id],
                    "recall": recall[class_id],
                    "dice": dice[class_id],
                    "iou": iou[class_id],
                }
            )

        total = int(matrix.sum().item())
        accuracy = tp.sum().item() / total if total else None

        return {
            "num_classes": self.num_classes,
            "total_pixels": total,
            "accuracy": accuracy,
            "macro_precision": macro(precision),
            "macro_recall": macro(recall),
            "macro_dice": macro(dice),
            "macro_iou": macro(iou),
            "per_class": per_class,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def reset(self):
        """Reset accumulated confusion matrix."""
        self.confusion_matrix.zero_()
