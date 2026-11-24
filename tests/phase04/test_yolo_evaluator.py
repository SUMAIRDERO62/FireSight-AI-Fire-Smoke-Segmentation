import pytest
import numpy as np

from firesight.evaluation.yolo_evaluator import (
    rasterize_yolo_labels,
    compute_mask_metrics,
)


def test_rasterize_preserves_fire_and_smoke_classes(tmp_path):
    label_file = tmp_path / "sample.txt"

    label_file.write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
        "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )

    mask = rasterize_yolo_labels(label_file, 100, 100, 2)

    assert mask.shape == (100, 100)
    assert mask[20, 20] == 1
    assert mask[70, 70] == 2
    assert 0 in np.unique(mask)
    assert 1 in np.unique(mask)
    assert 2 in np.unique(mask)


def test_missing_label_file_returns_background_mask(tmp_path):
    mask = rasterize_yolo_labels(
        tmp_path / "missing.txt", 64, 64, 2
    )

    assert mask.shape == (64, 64)
    assert np.all(mask == 0)


def test_empty_label_file_returns_background_mask(tmp_path):
    label_file = tmp_path / "empty.txt"
    label_file.write_text("")

    mask = rasterize_yolo_labels(label_file, 64, 64, 2)

    assert np.all(mask == 0)


def test_invalid_class_id_is_rejected(tmp_path):
    label_file = tmp_path / "invalid.txt"
    label_file.write_text("2 0.1 0.1 0.5 0.1 0.5 0.5\n")

    with pytest.raises(ValueError, match="Invalid class ID"):
        rasterize_yolo_labels(label_file, 100, 100, 2)


def test_invalid_polygon_is_rejected(tmp_path):
    label_file = tmp_path / "invalid_polygon.txt"
    label_file.write_text("0 0.1 0.1 0.5 0.5\n")

    with pytest.raises(ValueError, match="Invalid polygon"):
        rasterize_yolo_labels(label_file, 100, 100, 2)


def test_compute_mask_metrics():
    counts = [
        (80, 20, 20),
        (40, 10, 10),
    ]

    result = compute_mask_metrics(counts, ["fire", "smoke"])

    assert len(result["per_class"]) == 2

    fire = result["per_class"][0]
    smoke = result["per_class"][1]

    assert fire["class_name"] == "fire"
    assert fire["precision"] == pytest.approx(0.8)
    assert fire["recall"] == pytest.approx(0.8)
    assert fire["dice"] == pytest.approx(0.8)
    assert fire["iou"] == pytest.approx(80 / 120)

    assert smoke["class_name"] == "smoke"
    assert smoke["precision"] == pytest.approx(0.8)
    assert smoke["recall"] == pytest.approx(0.8)

    assert result["macro_precision"] == pytest.approx(0.8)
    assert result["macro_recall"] == pytest.approx(0.8)


def test_empty_class_metrics_return_none():
    result = compute_mask_metrics(
        [(0, 0, 0), (0, 0, 0)],
        ["fire", "smoke"],
    )

    assert result["macro_precision"] is None
    assert result["macro_recall"] is None
    assert result["macro_dice"] is None
    assert result["macro_iou"] is None
