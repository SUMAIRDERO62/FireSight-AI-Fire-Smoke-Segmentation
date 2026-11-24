"""Generate YOLO11-seg predictions and qualitative comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

from firesight.evaluation.yolo_evaluator import rasterize_yolo_labels


PALETTE = {
    0: (0, 0, 0),       # Background
    1: (255, 50, 50),   # Fire
    2: (255, 180, 0),   # Smoke
}


def resolve_dataset(data_yaml, split):
    data_yaml = Path(data_yaml).resolve()

    with data_yaml.open() as file:
        config = yaml.safe_load(file)

    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    image_dir = (root / config[split]).resolve()
    label_dir = root / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return image_dir, label_dir, config


def colorize(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color

    return Image.fromarray(rgb, mode="RGB")


def make_overlay(image, mask, alpha=0.45):
    return Image.blend(image.convert("RGB"), colorize(mask), alpha)


def create_comparison(image, ground_truth, prediction):
    original = image.convert("RGB")
    gt_overlay = make_overlay(original, ground_truth)
    pred_overlay = make_overlay(original, prediction)

    width, height = original.size
    canvas = Image.new("RGB", (width * 3, height + 35), "white")

    panels = [
        ("Original", original),
        ("Ground Truth", gt_overlay),
        ("YOLO Prediction", pred_overlay),
    ]

    draw = ImageDraw.Draw(canvas)

    for index, (label, panel) in enumerate(panels):
        x = index * width
        canvas.paste(panel, (x, 35))
        draw.text((x + 8, 10), label, fill="black")

    return canvas


def predict_image(model, image_path, image_size, confidence, device):
    result = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
    )[0]

    width, height = Image.open(image_path).size
    prediction = np.zeros((height, width), dtype=np.uint8)

    if result.masks is None or result.boxes is None:
        return prediction

    masks = result.masks.data
    classes = result.boxes.cls.int().cpu().tolist()

    masks = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(height, width),
        mode="nearest",
    ).squeeze(1) > 0.5

    for instance_mask, class_id in zip(masks, classes):
        if class_id not in (0, 1):
            continue

        prediction[instance_mask.cpu().numpy()] = class_id + 1

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO11-seg prediction visualizations."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default="data/processed/yolo11_seg_split_safe/data.yaml",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="runs/predictions/yolo11_seg",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    image_dir, label_dir, config = resolve_dataset(args.data, args.split)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(checkpoint))
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "split": args.split,
        "images_processed": 0,
        "classes": ["fire", "smoke"],
        "outputs": [],
    }

    for index, image_path in enumerate(image_paths, 1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = label_dir / f"{image_path.stem}.txt"
        ground_truth = rasterize_yolo_labels(
            label_path, width, height, num_classes=2
        )

        prediction = predict_image(
            model,
            image_path,
            args.imgsz,
            args.conf,
            args.device,
        )

        stem = image_path.stem

        save_dir = output_dir / stem
        save_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(prediction, mode="L").save(
            save_dir / "prediction_mask.png"
        )
        Image.fromarray(ground_truth, mode="L").save(
            save_dir / "ground_truth_mask.png"
        )

        make_overlay(image, prediction).save(
            save_dir / "prediction_overlay.png"
        )

        create_comparison(
            image, ground_truth, prediction
        ).save(save_dir / "comparison.png")

        summary["images_processed"] += 1
        summary["outputs"].append({
            "image": image_path.name,
            "output_directory": str(save_dir),
            "ground_truth_pixels": {
                "fire": int(np.sum(ground_truth == 1)),
                "smoke": int(np.sum(ground_truth == 2)),
            },
            "predicted_pixels": {
                "fire": int(np.sum(prediction == 1)),
                "smoke": int(np.sum(prediction == 2)),
            },
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"Processed: {image_path.name}"
        )

    report_path = output_dir / "prediction_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("\n========== PREDICTION COMPLETE ==========")
    print(json.dumps({
        "images_processed": summary["images_processed"],
        "output_directory": str(output_dir.resolve()),
        "summary_report": str(report_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
