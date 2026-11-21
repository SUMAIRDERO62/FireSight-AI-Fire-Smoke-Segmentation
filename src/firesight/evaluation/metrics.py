"""Segmentation metric utilities."""

import numpy as np


def dice_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate the Sørensen-Dice coefficient for binary masks."""
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction masks must have the same shape")
    denominator = int(target.sum()) + int(prediction.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(target, prediction).sum())
    return (2.0 * intersection) / denominator


def intersection_over_union(target: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate binary mask intersection over union."""
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction masks must have the same shape")
    union = int(np.logical_or(target, prediction).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(target, prediction).sum())
    return intersection / union

