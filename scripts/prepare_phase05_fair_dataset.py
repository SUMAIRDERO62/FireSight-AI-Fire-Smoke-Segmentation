"""Prepare fair U-Net and YOLO11-seg datasets from the verified COCO polygon source.

COCO category IDs: 1=fire, 2=smoke. Category 0 (fire-smoke) is intentionally ignored.
Both foreground classes are rasterized independently so overlapping pixels are retained.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/roboflow_fire_smoke_seg_v1"
OUTPUT = ROOT / "data/processed/phase05_fair"
SPLITS = {"train": "train", "valid": "val", "test": "test"}
CATEGORY_TO_CLASS = {1: 0, 2: 1}  # YOLO class IDs: fire=0, smoke=1


def polygon_points(segmentation, width, height):
    """Return valid COCO polygon coordinate lists."""
    if not isinstance(segmentation, list):
        raise ValueError("Expected COCO polygon segmentation list.")
    polygons = []
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        coords = []
        for x, y in zip(polygon[::2], polygon[1::2]):
            x = min(max(float(x), 0.0), float(width - 1))
            y = min(max(float(y), 0.0), float(height - 1))
            coords.append((x, y))
        polygons.append(coords)
    return polygons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()

    summary = {"source": str(source), "output": str(output), "splits": {}}
    for source_split, target_split in SPLITS.items():
        split_dir = source / source_split
        annotation_file = split_dir / "_annotations.coco.json"
        if not annotation_file.is_file():
            raise FileNotFoundError(annotation_file)
        coco = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = {int(c["id"]): c["name"] for c in coco["categories"]}
        if categories.get(1) != "fire" or categories.get(2) != "smoke":
            raise ValueError(f"Unexpected COCO categories: {categories}")

        images = {int(item["id"]): item for item in coco["images"]}
        annotations_by_image = {image_id: [] for image_id in images}
        for ann in coco["annotations"]:
            image_id = int(ann["image_id"])
            if image_id not in images:
                raise ValueError(f"Annotation references missing image id {image_id}")
            if int(ann["category_id"]) in CATEGORY_TO_CLASS:
                annotations_by_image[image_id].append(ann)

        unet_images = output / "unet" / target_split / "images"
        fire_masks = output / "unet" / target_split / "masks" / "fire"
        smoke_masks = output / "unet" / target_split / "masks" / "smoke"
        yolo_images = output / "yolo" / "images" / target_split
        yolo_labels = output / "yolo" / "labels" / target_split
        for directory in (unet_images, fire_masks, smoke_masks, yolo_images, yolo_labels):
            directory.mkdir(parents=True, exist_ok=True)

        annotation_count = {"fire": 0, "smoke": 0}
        overlap_pixels = 0
        for image_id, image_info in images.items():
            filename = Path(image_info["file_name"]).name
            source_image = split_dir / filename
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            with Image.open(source_image) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            if int(image_info["width"]) != width or int(image_info["height"]) != height:
                raise ValueError(f"COCO dimensions mismatch for {filename}")

            channels = [Image.new("L", (width, height), 0), Image.new("L", (width, height), 0)]
            draws = [ImageDraw.Draw(channels[0]), ImageDraw.Draw(channels[1])]
            yolo_lines = []
            for ann in annotations_by_image[image_id]:
                category_id = int(ann["category_id"])
                class_id = CATEGORY_TO_CLASS[category_id]
                class_name = "fire" if class_id == 0 else "smoke"
                annotation_count[class_name] += 1
                segmentation = ann.get("segmentation")
                for polygon in polygon_points(segmentation, width, height):
                    draws[class_id].polygon(polygon, fill=255)
                    normalized = []
                    for x, y in polygon:
                        normalized.extend((x / width, y / height))
                    yolo_lines.append(
                        str(class_id) + " " + " ".join(f"{v:.8f}" for v in normalized)
                    )

            fire_array = np.asarray(channels[0], dtype=np.uint8)
            smoke_array = np.asarray(channels[1], dtype=np.uint8)
            overlap_pixels += int(np.logical_and(fire_array > 0, smoke_array > 0).sum())

            # Save identical source image content into each model's split.
            image.save(unet_images / filename, quality=95)
            image.save(yolo_images / filename, quality=95)
            Image.fromarray(fire_array).save(fire_masks / f"{Path(filename).stem}.png")
            Image.fromarray(smoke_array).save(smoke_masks / f"{Path(filename).stem}.png")
            (yolo_labels / f"{Path(filename).stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

        summary["splits"][target_split] = {
            "images": len(images),
            "annotations": annotation_count,
            "overlap_pixels_sum_per_image": overlap_pixels,
        }

    yolo_yaml = output / "yolo" / "data.yaml"
    yolo_yaml.write_text(
        "path: " + str((output / "yolo").resolve()) + "\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: fire\n  1: smoke\n",
        encoding="utf-8",
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    expected = {"train": 141, "val": 40, "test": 20}
    for split, count in expected.items():
        actual = summary["splits"][split]["images"]
        if actual != count:
            raise RuntimeError(f"{split}: expected {count} images, got {actual}")
    print(json.dumps(summary, indent=2))
    print(f"\nYOLO YAML: {yolo_yaml}")
    print("Fair dataset preparation completed.")


if __name__ == "__main__":
    main()
