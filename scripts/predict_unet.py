"""Run U-Net segmentation on one image."""

import argparse
from pathlib import Path

import torch
import yaml

from firesight.evaluation.predictor import (
    load_image,
    save_prediction,
    save_overlay,
)
from firesight.models.unet import UNet


def main():
    parser = argparse.ArgumentParser(
        description="Generate a FireSight AI segmentation prediction."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--config", default="configs/unet.yaml", help="U-Net YAML configuration."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/unet_best.pth",
        help="Trained model checkpoint.",
    )
    parser.add_argument(
        "--output", default="outputs/predictions", help="Prediction output directory."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45, help="Overlay opacity between 0 and 1."
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    num_classes = config.get("num_classes")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise ValueError(
            "Set num_classes to a verified integer >= 2 before prediction."
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first or provide a valid checkpoint."
        )

    image_config = config["image"]
    model_config = config["model"]
    image_size = (image_config["height"], image_config["width"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_tensor, original_size = load_image(
        args.image,
        image_size,
        image_config["mean"],
        image_config["std"],
    )
    image_tensor = image_tensor.to(device)

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

    with torch.inference_mode():
        logits = model(image_tensor)
        prediction = logits.argmax(dim=1)[0].cpu()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(args.image).stem
    mask_path = output_dir / f"{image_stem}_mask.png"
    overlay_path = output_dir / f"{image_stem}_overlay.png"

    save_prediction(
        prediction,
        mask_path,
        num_classes,
        original_size=original_size,
    )
    save_overlay(
        args.image,
        prediction,
        overlay_path,
        num_classes,
        alpha=args.alpha,
    )

    print(f"Device: {device}")
    print(f"Prediction mask: {mask_path}")
    print(f"Overlay image: {overlay_path}")
    print("Prediction completed.")


if __name__ == "__main__":
    main()
