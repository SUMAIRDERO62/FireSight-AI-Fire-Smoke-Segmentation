from pathlib import Path

import pytest
import yaml

from firesight.training.yolo_trainer import (
    load_training_config,
    resolve_dataset,
)


def make_dataset(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)

    dataset_yaml = root / "data.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump({
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "fire", 1: "smoke"},
        }),
        encoding="utf-8",
    )
    return dataset_yaml


def test_load_project_config():
    root = Path(__file__).resolve().parents[2]
    config = load_training_config(root / "configs/yolo11_seg.yaml")
    assert config["model"]["task"] == "segment"
    assert config["training"]["epochs"] == 100


def test_resolve_dataset_paths(tmp_path):
    dataset_yaml = make_dataset(tmp_path)
    result = resolve_dataset(dataset_yaml)

    assert Path(result["train"]).is_dir()
    assert Path(result["val"]).is_dir()
    assert Path(result["test"]).is_dir()
    assert result["nc"] == 2
    assert result["names"] == {0: "fire", 1: "smoke"}


def test_reject_wrong_class_names(tmp_path):
    dataset_yaml = make_dataset(tmp_path)
    data = yaml.safe_load(dataset_yaml.read_text())
    data["names"] = {0: "flame", 1: "smoke"}
    dataset_yaml.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="Expected classes"):
        resolve_dataset(dataset_yaml)


def test_reject_missing_validation_split(tmp_path):
    dataset_yaml = make_dataset(tmp_path)
    data = yaml.safe_load(dataset_yaml.read_text())
    del data["val"]
    dataset_yaml.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="val"):
        resolve_dataset(dataset_yaml)


def test_reject_missing_dataset_yaml(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_dataset(tmp_path / "missing.yaml")


def test_reject_missing_config_section(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("model: {}\n")

    with pytest.raises(ValueError, match="dataset"):
        load_training_config(config)
