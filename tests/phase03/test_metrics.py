import pytest
import torch

from firesight.evaluation.metrics import SegmentationMetrics


def test_perfect_predictions_have_perfect_metrics():
    targets = torch.tensor([[[0, 1], [1, 0]]])
    metrics = SegmentationMetrics(2)
    metrics.update(targets, targets)

    result = metrics.compute()

    assert result["accuracy"] == 1.0
    assert result["macro_dice"] == 1.0
    assert result["macro_iou"] == 1.0
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 1.0


def test_confusion_matrix_counts_are_correct():
    targets = torch.tensor([[[0, 0], [1, 1]]])
    predictions = torch.tensor([[[0, 1], [1, 0]]])

    metrics = SegmentationMetrics(2)
    metrics.update(predictions, targets)

    result = metrics.compute()

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["total_pixels"] == 4
    assert result["accuracy"] == 0.5


def test_logits_are_converted_to_class_predictions():
    targets = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.tensor(
        [
            [
                [[5.0, 1.0], [1.0, 5.0]],
                [[1.0, 5.0], [5.0, 1.0]],
            ]
        ]
    )

    metrics = SegmentationMetrics(2)
    metrics.update(logits, targets)

    assert metrics.compute()["accuracy"] == 1.0


def test_ignore_index_is_excluded():
    targets = torch.tensor([[[0, 1], [255, 0]]])
    predictions = torch.tensor([[[0, 0], [1, 0]]])

    metrics = SegmentationMetrics(2, ignore_index=255)
    metrics.update(predictions, targets)

    result = metrics.compute()

    assert result["total_pixels"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_metrics_accumulate_multiple_batches():
    metrics = SegmentationMetrics(2)

    targets = torch.tensor([[[0, 1]]])
    metrics.update(targets, targets)

    targets2 = torch.tensor([[[1, 0]]])
    metrics.update(targets2, targets2)

    assert metrics.compute()["total_pixels"] == 4
    assert metrics.compute()["accuracy"] == 1.0


def test_invalid_target_class_is_rejected():
    metrics = SegmentationMetrics(2)

    with pytest.raises(ValueError, match="Targets"):
        metrics.update(
            torch.tensor([[[0, 1]]]),
            torch.tensor([[[0, 3]]]),
        )


def test_reset_clears_confusion_matrix():
    metrics = SegmentationMetrics(2)
    targets = torch.tensor([[[0, 1]]])
    metrics.update(targets, targets)
    metrics.reset()

    result = metrics.compute()

    assert result["total_pixels"] == 0
    assert result["accuracy"] is None


def test_invalid_class_count_is_rejected():
    with pytest.raises(ValueError):
        SegmentationMetrics(1)
