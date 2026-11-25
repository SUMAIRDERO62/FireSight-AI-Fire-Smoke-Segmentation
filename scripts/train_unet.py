"""Train the two-channel FireSight U-Net."""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from firesight.data.multilabel_dataset import MultiLabelSegmentationDataset
from firesight.data.transforms import SegmentationTransform
from firesight.models.unet import UNet
from firesight.training.losses import BCEWithLogitsDiceLoss
from firesight.training.trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    def project_path(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    seed = int(config.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = config["data"]
    image_cfg, aug = config["image"], config["augmentation"]
    size = (int(image_cfg["height"]), int(image_cfg["width"]))
    def make_dataset(split, training):
        base = project_path(data[f"{split}_root"])
        transform = SegmentationTransform(
            mean=tuple(image_cfg["mean"]), std=tuple(image_cfg["std"]),
            horizontal_flip_prob=float(aug["horizontal_flip_prob"]) if training else 0.0,
            vertical_flip_prob=float(aug["vertical_flip_prob"]) if training else 0.0,
        )
        return MultiLabelSegmentationDataset(
            base / "images", base / "masks/fire", base / "masks/smoke",
            image_size=size, transform=transform,
        )

    train_ds, val_ds = make_dataset("train", True), make_dataset("val", False)
    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(batch_size=int(train_cfg["batch_size"]), num_workers=int(train_cfg["num_workers"]),
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)

    model = UNet(num_classes=2, in_channels=int(config["model"]["in_channels"]),
                 base_channels=int(config["model"]["base_channels"]))
    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["learning_rate"]),
                                  weight_decay=float(opt_cfg["weight_decay"]))
    sched_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]), min_lr=float(sched_cfg["min_learning_rate"])
    ) if sched_cfg["enabled"] else None
    loss_cfg = config["loss"]
    criterion = BCEWithLogitsDiceLoss(
        bce_weight=float(loss_cfg["bce_weight"]),
        dice_weight=float(loss_cfg["dice_weight"]),
    )
    output = config["output"]
    trainer = SegmentationTrainer(
        model, optimizer, criterion, device=device,
        checkpoint_dir=project_path(output["checkpoint_dir"]),
        gradient_clip=float(train_cfg["gradient_clip"]), scheduler=scheduler,
    )
    start = trainer.resume() if args.resume else 1
    print(f"Device: {device}; train={len(train_ds)}; val={len(val_ds)}; classes=fire,smoke")
    trainer.fit(
        train_loader, val_loader, epochs=int(train_cfg["epochs"]),
        start_epoch=start,
        early_stopping_patience=int(train_cfg["early_stopping_patience"]),
        min_delta=float(train_cfg["min_delta"]),
        history_dir=project_path(output["history_dir"]),
    )


if __name__ == "__main__":
    main()
