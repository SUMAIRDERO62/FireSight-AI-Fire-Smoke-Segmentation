"""Configuration-driven YOLO11 segmentation training."""

from pathlib import Path
from typing import Any

import yaml


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the project training configuration."""
    config_path = Path(config_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    for section in ("model", "dataset", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")

    return config


def resolve_dataset(dataset_yaml: str | Path) -> dict[str, Any]:
    """Resolve dataset splits to absolute paths and validate them."""
    dataset_yaml = Path(dataset_yaml).resolve()

    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping.")

    root_value = Path(data.get("path", "."))
    root = root_value if root_value.is_absolute() else dataset_yaml.parent / root_value
    root = root.resolve()

    names = data.get("names")
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        raise ValueError("Dataset must define class names.")

    if names != {0: "fire", 1: "smoke"}:
        raise ValueError(f"Expected classes fire and smoke; found: {names}")

    resolved = {
        "path": str(root),
        "names": names,
        "nc": len(names),
    }

    for split in ("train", "val"):
        if split not in data:
            raise ValueError(f"Dataset YAML is missing '{split}'.")

        split_path = Path(data[split])
        split_path = split_path if split_path.is_absolute() else root / split_path
        split_path = split_path.resolve()

        if not split_path.is_dir():
            raise FileNotFoundError(f"{split} images directory not found: {split_path}")

        resolved[split] = str(split_path)

    if "test" in data:
        test_path = Path(data["test"])
        test_path = test_path if test_path.is_absolute() else root / test_path
        test_path = test_path.resolve()

        if not test_path.is_dir():
            raise FileNotFoundError(f"Test images directory not found: {test_path}")

        resolved["test"] = str(test_path)

    return resolved


class YOLOSegTrainer:
    """Run reproducible YOLO segmentation training."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config = load_training_config(self.config_path)

        project_root = self.config_path.parent.parent
        dataset_yaml = project_root / self.config["dataset"]["yaml"]
        self.dataset = resolve_dataset(dataset_yaml)

    def train(
        self,
        epochs: int | None = None,
        device: str | None = None,
        imgsz: int | None = None,
        batch: int | None = None,
        workers: int | None = None,
    ):
        from firesight.models.yolo_seg import build_yolo_seg_model

        model_config = self.config["model"]
        training_config = dict(self.config["training"])

        if epochs is not None:
            if epochs < 1:
                raise ValueError("epochs must be at least 1")
            training_config["epochs"] = epochs

        if device is not None:
            training_config["device"] = device
        if imgsz is not None:
            training_config["imgsz"] = imgsz
        if batch is not None:
            training_config["batch"] = batch
        if workers is not None:
            training_config["workers"] = workers

        # Keep output paths anchored to the project, not the shell's cwd.
        project_root = self.config_path.parent.parent
        output_project = Path(training_config.get("project", "runs/segment"))
        if not output_project.is_absolute():
            output_project = project_root / output_project

        training_config["project"] = str(output_project)

        model = build_yolo_seg_model(model_config["weights"])

        dataset_yaml = (
            self.config_path.parent.parent
            / self.config["dataset"]["yaml"]
        ).resolve()

        return model.train(
            data=str(dataset_yaml),
            **training_config,
        )
