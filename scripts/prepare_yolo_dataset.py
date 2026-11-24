from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path


SOURCE_DEFAULT = Path("data/raw/roboflow_fire_smoke_seg_v1")
OUTPUT_DEFAULT = Path("data/processed/yolo11_seg")

# COCO category IDs -> YOLO class IDs
CATEGORY_MAP = {
    1: 0,  # fire
    2: 1,  # smoke
}

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


class DatasetValidationError(Exception):
    pass


def resolve_image(source_root: Path, split: str, file_name: str) -> Path:
    relative = Path(file_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetValidationError(
            f"Unsafe image path in COCO annotation: {file_name}"
        )

    candidates = [
        source_root / split / relative,
        source_root / relative,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise DatasetValidationError(
        f"Image file not found: split={split}, file={file_name}"
    )


def polygon_area(points: list[float]) -> float:
    coordinates = list(zip(points[0::2], points[1::2]))
    area = 0.0

    for i, (x1, y1) in enumerate(coordinates):
        x2, y2 = coordinates[(i + 1) % len(coordinates)]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def validate_and_prepare(source_root: Path) -> tuple[dict, dict]:
    prepared = {}
    report = {
        "source": str(source_root),
        "classes": CLASS_NAMES,
        "splits": {},
        "total_images": 0,
        "total_annotations": 0,
        "warnings": [],
    }

    for source_split, output_split in SPLITS.items():
        json_path = (
            source_root / source_split / "_annotations.coco.json"
        )

        if not json_path.is_file():
            raise DatasetValidationError(
                f"COCO annotation file missing: {json_path}"
            )

        data = json.loads(json_path.read_text(encoding="utf-8"))

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        category_ids = {item["id"] for item in categories}
        if not set(CATEGORY_MAP).issubset(category_ids):
            raise DatasetValidationError(
                f"Expected fire/smoke categories missing in {source_split}"
            )

        image_map = {}
        image_paths = {}
        output_names = set()
        output_label_names = set()

        for image in images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            width = image.get("width")
            height = image.get("height")

            if image_id in image_map:
                raise DatasetValidationError(
                    f"Duplicate image ID: {image_id} in {source_split}"
                )

            if not isinstance(width, int) or not isinstance(height, int):
                raise DatasetValidationError(
                    f"Invalid dimensions for image: {file_name}"
                )

            if width <= 0 or height <= 0:
                raise DatasetValidationError(
                    f"Non-positive image dimensions: {file_name}"
                )

            if not file_name:
                raise DatasetValidationError(
                    f"Image {image_id} has no file_name"
                )

            source_image = resolve_image(
                source_root, source_split, file_name
            )

            output_name = Path(file_name).name
            label_name = Path(output_name).stem + ".txt"

            if output_name.lower() in output_names:
                raise DatasetValidationError(
                    f"Duplicate output image filename: {output_name}"
                )

            if label_name.lower() in output_label_names:
                raise DatasetValidationError(
                    f"Duplicate output label filename: {label_name}"
                )

            output_names.add(output_name.lower())
            output_label_names.add(label_name.lower())

            image_map[image_id] = image
            image_paths[image_id] = source_image

        labels = {image_id: [] for image_id in image_map}
        class_counts = Counter()
        annotation_ids = set()

        for annotation in annotations:
            annotation_id = annotation.get("id")

            if annotation_id in annotation_ids:
                raise DatasetValidationError(
                    f"Duplicate annotation ID: {annotation_id}"
                )
            annotation_ids.add(annotation_id)

            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")

            if image_id not in image_map:
                raise DatasetValidationError(
                    f"Annotation {annotation_id} references unknown image"
                )

            if category_id not in CATEGORY_MAP:
                raise DatasetValidationError(
                    f"Unknown category {category_id} in {source_split}"
                )

            segmentation = annotation.get("segmentation")

            if not isinstance(segmentation, list) or len(segmentation) != 1:
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: expected exactly one polygon"
                )

            polygon = segmentation[0]

            if not isinstance(polygon, list) or len(polygon) < 6:
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: polygon has too few coordinates"
                )

            if len(polygon) % 2 != 0:
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: odd coordinate count"
                )

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in polygon
            ):
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: invalid coordinate value"
                )

            image = image_map[image_id]
            width = image["width"]
            height = image["height"]

            xs = polygon[0::2]
            ys = polygon[1::2]

            if any(x < 0 or x > width for x in xs):
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: x coordinate out of bounds"
                )

            if any(y < 0 or y > height for y in ys):
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: y coordinate out of bounds"
                )

            if polygon_area(polygon) <= 0:
                raise DatasetValidationError(
                    f"Annotation {annotation_id}: zero-area polygon"
                )

            class_id = CATEGORY_MAP[category_id]

            normalized = []
            for x, y in zip(xs, ys):
                normalized.extend([
                    x / width,
                    y / height,
                ])

            line = str(class_id) + " " + " ".join(
                f"{value:.8f}" for value in normalized
            )

            labels[image_id].append(line)
            class_counts[CLASS_NAMES[class_id]] += 1

        prepared[source_split] = {
            "output_split": output_split,
            "images": image_map,
            "image_paths": image_paths,
            "labels": labels,
        }

        split_report = {
            "images": len(images),
            "annotations": len(annotations),
            "class_counts": dict(class_counts),
            "empty_label_images": sum(
                1 for lines in labels.values() if not lines
            ),
        }

        report["splits"][output_split] = split_report
        report["total_images"] += len(images)
        report["total_annotations"] += len(annotations)

    return prepared, report


def write_dataset(
    prepared: dict,
    report: dict,
    output_root: Path,
    overwrite: bool = False,
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)

    if output_root.exists():
        existing_files = [
            item for item in output_root.rglob("*")
            if item.is_file()
        ]
        if existing_files and not overwrite:
            raise DatasetValidationError(
                f"Output contains existing files: {output_root}. "
                "Use --overwrite only if you intentionally want to replace them."
            )

    staging = Path(tempfile.mkdtemp(
        prefix=".yolo11_seg_staging_",
        dir=output_root.parent,
    ))

    try:
        for split_data in prepared.values():
            split = split_data["output_split"]

            image_dir = staging / "images" / split
            label_dir = staging / "labels" / split

            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            for image_id, image in split_data["images"].items():
                source_image = split_data["image_paths"][image_id]
                image_name = Path(image["file_name"]).name

                shutil.copy2(
                    source_image,
                    image_dir / image_name,
                )

                label_name = Path(image_name).stem + ".txt"
                label_content = "\n".join(
                    split_data["labels"][image_id]
                )

                if label_content:
                    label_content += "\n"

                (label_dir / label_name).write_text(
                    label_content,
                    encoding="utf-8",
                )

        report["format"] = "YOLO segmentation"
        report["class_names"] = CLASS_NAMES

        (staging / "dataset_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        lines = [
            "FIRESIGHT AI - YOLO SEGMENTATION DATASET REPORT",
            "=" * 55,
            f"Total images: {report['total_images']}",
            f"Total annotations: {report['total_annotations']}",
            "",
        ]

        for split, stats in report["splits"].items():
            lines.extend([
                split.upper(),
                f"Images: {stats['images']}",
                f"Annotations: {stats['annotations']}",
                f"Class counts: {stats['class_counts']}",
                f"Empty label images: {stats['empty_label_images']}",
                "",
            ])

        lines.extend([
            "CLASS MAPPING",
            "0 = fire",
            "1 = smoke",
            "",
            "Original COCO files were not modified.",
        ])

        (staging / "dataset_report.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        if output_root.exists():
            existing_files = [
                item for item in output_root.rglob("*")
                if item.is_file()
            ]
            if existing_files and not overwrite:
                raise DatasetValidationError(
                    "Output contains files; refusing to replace them."
                )
            shutil.rmtree(output_root)

        staging.rename(output_root)

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert FireSight COCO polygons to YOLO segmentation."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DEFAULT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )

    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"Source dataset not found: {source_root}")

    try:
        print("=" * 65)
        print("FIRESIGHT AI - COCO TO YOLO SEGMENTATION")
        print("=" * 65)

        print("\nSTEP 1: Validating all source annotations...")
        prepared, report = validate_and_prepare(source_root)

        print("Validation successful.")
        print(f"Images: {report['total_images']}")
        print(f"Annotations: {report['total_annotations']}")

        print("\nSTEP 2: Writing YOLO dataset...")
        write_dataset(
            prepared,
            report,
            output_root,
            overwrite=args.overwrite,
        )

        print("\nSTEP 3: Conversion summary")
        for split, stats in report["splits"].items():
            print(
                f"{split}: "
                f"{stats['images']} images, "
                f"{stats['annotations']} annotations, "
                f"{stats['empty_label_images']} empty labels"
            )

        print("\nOUTPUT:", output_root)
        print("Reports: dataset_report.json, dataset_report.txt")
        print("\nCONVERSION COMPLETED SUCCESSFULLY.")

    except (DatasetValidationError, json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"\nCONVERSION FAILED: {error}")


if __name__ == "__main__":
    main()
