from pathlib import Path
from collections import Counter
import json
import math
import sys

DATASET = Path("data/processed/yolo11_seg")

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "fire", 1: "smoke"}

report = {
    "dataset": str(DATASET),
    "splits": {},
    "errors": [],
    "warnings": [],
}

total_images = 0
total_polygons = 0
total_classes = Counter()


def error(message):
    report["errors"].append(message)
    print("[ERROR]", message)


def warning(message):
    report["warnings"].append(message)
    print("[WARNING]", message)


print("=" * 65)
print("FIRESIGHT AI - INDEPENDENT YOLO DATASET AUDIT")
print("=" * 65)

if not DATASET.is_dir():
    raise SystemExit(f"Dataset directory not found: {DATASET}")

for split in SPLITS:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    if not image_dir.is_dir() or not label_dir.is_dir():
        error(f"Missing image or label directory: {split}")
        continue

    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem.lower() for p in images}
    label_stems = {p.stem.lower() for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:
        error(f"{split}: {len(missing_labels)} images without label files")

    if orphan_labels:
        error(f"{split}: {len(orphan_labels)} labels without images")

    class_counts = Counter()
    polygon_count = 0
    empty_labels = 0
    checked_lines = 0

    for label_path in labels:
        content = label_path.read_text(encoding="utf-8").strip()

        if not content:
            empty_labels += 1
            continue

        for line_number, line in enumerate(content.splitlines(), 1):
            checked_lines += 1
            parts = line.split()

            if len(parts) < 7:
                error(
                    f"{split}/{label_path.name}:{line_number}: "
                    "too few values for segmentation polygon"
                )
                continue

            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                error(
                    f"{split}/{label_path.name}:{line_number}: non-numeric label value"
                )
                continue

            if class_id not in CLASS_NAMES:
                error(
                    f"{split}/{label_path.name}:{line_number}: "
                    f"unknown class ID {class_id}"
                )
                continue

            if len(coordinates) % 2 != 0:
                error(f"{split}/{label_path.name}:{line_number}: odd coordinate count")
                continue

            if len(coordinates) < 6:
                error(
                    f"{split}/{label_path.name}:{line_number}: "
                    "polygon has fewer than 3 points"
                )
                continue

            if not all(math.isfinite(v) for v in coordinates):
                error(f"{split}/{label_path.name}:{line_number}: non-finite coordinate")
                continue

            if any(v < 0 or v > 1 for v in coordinates):
                error(
                    f"{split}/{label_path.name}:{line_number}: "
                    "coordinate outside normalized range [0, 1]"
                )
                continue

            class_counts[CLASS_NAMES[class_id]] += 1
            polygon_count += 1

    total_images += len(images)
    total_polygons += polygon_count
    total_classes.update(class_counts)

    if empty_labels:
        warning(f"{split}: {empty_labels} empty label files")

    report["splits"][split] = {
        "images": len(images),
        "label_files": len(labels),
        "polygons": polygon_count,
        "empty_labels": empty_labels,
        "class_counts": dict(class_counts),
        "checked_label_lines": checked_lines,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
    }

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Polygons:", polygon_count)
    print("Empty labels:", empty_labels)
    print("Class counts:", dict(class_counts))
    print("Checked label lines:", checked_lines)

print("\n" + "=" * 65)
print("AUDIT SUMMARY")
print("=" * 65)
print("Total images:", total_images)
print("Total polygons:", total_polygons)
print("Total class counts:", dict(total_classes))
print("Errors:", len(report["errors"]))
print("Warnings:", len(report["warnings"]))

report["total_images"] = total_images
report["total_polygons"] = total_polygons
report["total_class_counts"] = dict(total_classes)

report_path = DATASET / "independent_audit.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nAudit report:", report_path)

if report["errors"]:
    print("\nAUDIT FAILED. Fix errors before training.")
    sys.exit(1)

print("\nAUDIT PASSED.")
