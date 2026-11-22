"""Evaluate a trained FireSight AI U-Net segmentation model."""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from firesight.data.segmentation_dataset import SegmentationDataset
from firesight.data.transforms import SegmentationTransform
from firesight.evaluation.metrics import SegmentationMetrics
from firesight.models.unet import UNet


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML dictionary.")

    return config


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_metrics.json"
    csv_path = output_dir / "per_class_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    fields = [
        "class_id",
        "support",
        "predicted_pixels",
        "precision",
        "recall",
        "dice",
        "iou",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results["per_class"])

    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FireSight AI U-Net model."
    )
    parser.add_argument(
        "--config",
        default="configs/unet.yaml",
        help="Path to the U-Net YAML configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/unet_best.pth",
        help="Path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Optional override for the evaluation images directory.",
    )
    parser.add_argument(
        "--masks",
        default=None,
        help="Optional override for the ground-truth masks directory.",
    )
    parser.add_argument(
        "--output",
        default="reports/evaluation",
        help="Directory for evaluation reports.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    num_classes = config.get("num_classes")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise ValueError(
            "Set num_classes in configs/unet.yaml to the verified "
            "number of segmentation classes before evaluation."
        )

    data_config = config["data"]
    image_config = config["image"]
    model_config = config["model"]
    training_config = config["training"]

    image_key = f"{args.split}_images"
    mask_key = f"{args.split}_masks"

    images_dir = args.images or data_config.get(image_key)
    masks_dir = args.masks or data_config.get(mask_key)

    if not images_dir or not masks_dir:
        raise ValueError(
            f"Configure {image_key} and {mask_key}, or provide --images and --masks."
        )

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the model first or provide --checkpoint."
        )

    image_size = (
        image_config["height"],
        image_config["width"],
    )

    transform = SegmentationTransform(
        image_size=image_size,
        mean=tuple(image_config["mean"]),
        std=tuple(image_config["std"]),
    )

    dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        num_classes=num_classes,
        transform=transform,
    )

    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    batch_size = int(training_config["batch_size"])
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(
        num_classes=num_classes,
        in_channels=model_config["in_channels"],
        base_channels=model_config["base_channels"],
    ).to(device)

    # Load only trusted checkpoints created by this project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ignore_index = config.get("loss", {}).get("ignore_index")
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Evaluation images: {len(dataset)}")
    print(f"Classes: {num_classes}")
    print("Evaluating...")

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            logits = model(images)
            metrics.update(logits, masks)

    results = metrics.compute()
    results["split"] = args.split
    results["checkpoint"] = str(checkpoint_path)
    results["images_evaluated"] = len(dataset)

    save_results(results, args.output)

    print("\nEvaluation summary")
    print("------------------")
    for key in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_dice",
        "macro_iou",
    ):
        value = results[key]
        display = f"{value:.4f}" if value is not None else "N/A"
        print(f"{key}: {display}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
