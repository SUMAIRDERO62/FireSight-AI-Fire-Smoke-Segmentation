"""Evaluation utilities for YOLO instance-segmentation models."""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw


def load_dataset_info(data_yaml, split="test"):
    """Resolve dataset image and label directories from YOLO YAML."""
    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    data = yaml.safe_load(data_yaml.read_text())
    if split not in data:
        raise ValueError(f"Dataset split missing: {split}")

    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    image_dir = Path(data[split])
    if not image_dir.is_absolute():
        image_dir = root / image_dir

    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory missing: {image_dir}")

    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    return data, image_dir, label_dir


def rasterize_yolo_labels(label_path, width, height, num_classes):
    """Convert normalized YOLO polygon labels into class-ID masks."""
    mask = np.zeros((height, width), dtype=np.uint8)
    label_path = Path(label_path)

    if not label_path.exists():
        return mask

    image = Image.fromarray(mask)
    draw = ImageDraw.Draw(image)

    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue

        values = [float(v) for v in parts]
        class_id = int(values[0])

        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"Invalid class ID at {label_path}:{line_number}")

        coords = values[1:]
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"Invalid polygon at {label_path}:{line_number}")

        points = []
        for x, y in zip(coords[::2], coords[1::2]):
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Non-normalized polygon at {label_path}:{line_number}")
            points.append((round(x * (width - 1)), round(y * (height - 1))))

        draw.polygon(points, fill=class_id + 1)

    result = np.asarray(image, dtype=np.uint8)
    return result


def compute_mask_metrics(counts, class_names):
    """Calculate aggregate pixel-level metrics for each foreground class."""
    per_class = []

    for class_id, name in enumerate(class_names):
        tp, fp, fn = counts[class_id]

        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        iou = tp / (tp + fp + fn) if tp + fp + fn else None

        per_class.append({
            "class_id": class_id,
            "class_name": str(name),
            "tp_pixels": int(tp),
            "fp_pixels": int(fp),
            "fn_pixels": int(fn),
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
        })

    def macro(key):
        values = [item[key] for item in per_class if item[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_dice": macro("dice"),
        "macro_iou": macro("iou"),
        "per_class": per_class,
    }


class YOLOSegEvaluator:
    """Evaluate YOLO segmentation using native AP and pixel-level metrics."""

    def __init__(
        self,
        checkpoint,
        data_yaml,
        split="test",
        imgsz=640,
        device=None,
        conf=0.25,
    ):
        self.checkpoint = Path(checkpoint).resolve()
        self.data_yaml = Path(data_yaml).resolve()
        self.split = split
        self.imgsz = imgsz
        self.device = device
        self.conf = conf

        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

        self.data, self.image_dir, self.label_dir = load_dataset_info(
            self.data_yaml, split
        )
        self.names = self.data["names"]
        self.num_classes = len(self.names)

        if isinstance(self.names, dict):
            self.class_names = [
                str(self.names.get(i, self.names.get(str(i))))
                for i in range(self.num_classes)
            ]
        else:
            self.class_names = list(self.names)

        if self.num_classes < 1:
            raise ValueError("Dataset must contain at least one class.")

    @staticmethod
    def _metric_value(metrics, key):
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)

    @torch.no_grad()
    def evaluate(self, output_dir="runs/evaluation/yolo11_seg"):
        from ultralytics import YOLO

        model = YOLO(str(self.checkpoint))

        # Official Ultralytics detection and instance-segmentation metrics.
        native = model.val(
            data=str(self.data_yaml),
            split=self.split,
            imgsz=self.imgsz,
            device=self.device,
            plots=True,
            verbose=False,
        )

        native_values = native.results_dict

        image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        # Counts are accumulated across all pixels and images.
        counts = [[0, 0, 0] for _ in range(self.num_classes)]

        for image_path in image_paths:
            with Image.open(image_path) as im:
                width, height = im.size

            gt = rasterize_yolo_labels(
                self.label_dir / f"{image_path.stem}.txt",
                width,
                height,
                self.num_classes,
            )

            predictions = model.predict(
                source=str(image_path),
                imgsz=self.imgsz,
                device=self.device,
                conf=self.conf,
                verbose=False,
            )
            result = predictions[0]

            pred = np.zeros((height, width), dtype=np.uint8)

            if result.masks is not None and result.boxes is not None:
                mask_data = result.masks.data.float()
                mask_data = F.interpolate(
                    mask_data.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0] > 0.5

                class_ids = result.boxes.cls.int().cpu().tolist()

                for instance_mask, class_id in zip(
                    mask_data.cpu().numpy(), class_ids
                ):
                    if 0 <= class_id < self.num_classes:
                        pred[instance_mask] = class_id + 1

            for class_id in range(self.num_classes):
                target = gt == (class_id + 1)
                predicted = pred == (class_id + 1)

                tp = int(np.logical_and(target, predicted).sum())
                fp = int(np.logical_and(~target, predicted).sum())
                fn = int(np.logical_and(target, ~predicted).sum())

                counts[class_id][0] += tp
                counts[class_id][1] += fp
                counts[class_id][2] += fn

        mask_metrics = compute_mask_metrics(counts, self.class_names)

        report = {
            "checkpoint": str(self.checkpoint),
            "dataset_yaml": str(self.data_yaml),
            "split": self.split,
            "images_evaluated": len(image_paths),
            "image_size": self.imgsz,
            "confidence_threshold": self.conf,
            "classes": self.class_names,
            "native_yolo_metrics": {
                key: self._metric_value(native_values, key)
                for key in (
                    "metrics/precision(B)",
                    "metrics/recall(B)",
                    "metrics/mAP50(B)",
                    "metrics/mAP50-95(B)",
                    "metrics/precision(M)",
                    "metrics/recall(M)",
                    "metrics/mAP50(M)",
                    "metrics/mAP50-95(M)",
                )
            },
            "pixel_segmentation_metrics": mask_metrics,
            "note": (
                "Pixel metrics are aggregate foreground-class metrics. "
                "Native YOLO AP metrics are calculated independently."
            ),
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        report["report_path"] = str(report_path.resolve())
        return report
