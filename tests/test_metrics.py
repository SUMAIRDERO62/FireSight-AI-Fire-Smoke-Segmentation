"""Tests for segmentation metrics."""

import numpy as np

from firesight.evaluation.metrics import dice_score, intersection_over_union


def test_perfect_masks_score_one():
    mask = np.ones((4, 4), dtype=bool)
    assert dice_score(mask, mask) == 1.0
    assert intersection_over_union(mask, mask) == 1.0


def test_empty_masks_score_one():
    mask = np.zeros((4, 4), dtype=bool)
    assert dice_score(mask, mask) == 1.0
    assert intersection_over_union(mask, mask) == 1.0


def test_partially_overlapping_masks():
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    assert dice_score(target, prediction) == 0.5
    assert intersection_over_union(target, prediction) == 0.3333333333333333
