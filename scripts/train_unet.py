"""Train FireSight AI U-Net using a YAML configuration."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from firesight.data.segmentation_dataset import SegmentationDataset
from firesight.data.transforms import SegmentationTransform
from firesight.models.unet import UNet
from firesight.training.losses import CrossEntropyDiceLoss
from firesight.training.trainer import SegmentationTrainer


def load_config(config_path):
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_directory(path, description):
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(f"{description} directory not found: {path}")

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Train the FireSight AI U-Net segmentation model."
    )
    parser.add_argument(
        "--config",
        default="configs/unet.yaml",
        help="Path to the YAML training configuration.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    num_classes = config.get("num_classes")

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            "Set num_classes to a verified integer >= 2 in the "
            "configuration. Verify genuine mask class IDs first."
        )

    seed = config.get("seed", 42)
    set_seed(seed)

    data_config = config["data"]
    image_config = config["image"]
    training_config = config["training"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    loss_config = config["loss"]
    output_config = config["output"]

    train_images = require_directory(data_config["train_images"], "Training images")
    train_masks = require_directory(data_config["train_masks"], "Training masks")
    val_images = require_directory(data_config["val_images"], "Validation images")
    val_masks = require_directory(data_config["val_masks"], "Validation masks")

    image_size = (
        int(image_config["height"]),
        int(image_config["width"]),
    )

    if min(image_size) <= 0:
        raise ValueError("Image dimensions must be positive.")

    train_transform = SegmentationTransform(
        image_size=image_size,
        mean=tuple(image_config["mean"]),
        std=tuple(image_config["std"]),
        horizontal_flip_prob=config["augmentation"]["horizontal_flip_prob"],
        vertical_flip_prob=config["augmentation"]["vertical_flip_prob"],
    )

    val_transform = SegmentationTransform(
        image_size=image_size,
        mean=tuple(image_config["mean"]),
        std=tuple(image_config["std"]),
    )

    train_dataset = SegmentationDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        num_classes=num_classes,
        transform=train_transform,
    )

    val_dataset = SegmentationDataset(
        images_dir=val_images,
        masks_dir=val_masks,
        num_classes=num_classes,
        transform=val_transform,
    )

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])

    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid batch size or worker count.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(
        num_classes=num_classes,
        in_channels=int(model_config["in_channels"]),
        base_channels=int(model_config["base_channels"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    scheduler_config = config["scheduler"]
    scheduler = None

    if scheduler_config["enabled"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_learning_rate"]),
        )

    criterion = CrossEntropyDiceLoss(
        ce_weight=float(loss_config["cross_entropy_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
        ignore_index=loss_config["ignore_index"],
    )

    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=output_config["checkpoint_dir"],
        gradient_clip=training_config["gradient_clip"],
        scheduler=scheduler,
    )

    start_epoch = 1

    if args.resume:
        start_epoch = trainer.resume()
        print(f"Resuming training from epoch {start_epoch}.")

    print(f"Device: {device}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(training_config["epochs"]),
        start_epoch=start_epoch,
        early_stopping_patience=int(training_config["early_stopping_patience"]),
        min_delta=float(training_config["min_delta"]),
        history_dir=output_config["history_dir"],
    )


if __name__ == "__main__":
    main()
