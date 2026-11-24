import csv
import shutil
from pathlib import Path
from collections import Counter

source = Path("data/processed/yolo11_seg")
destination = Path("data/processed/yolo11_seg_split_safe")
pairs_csv = Path("reports/results/phase04_near_duplicate_pairs.csv")

if not source.is_dir():
    raise SystemExit("ERROR: Original processed dataset not found.")
if not pairs_csv.is_file():
    raise SystemExit("ERROR: Duplicate candidate CSV not found.")
if destination.exists():
    raise SystemExit(
        f"STOP: {destination} already exists. Original files will not be overwritten."
    )

with pairs_csv.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Collect unique validation/test counterparts from the 10 v1 cross-split pairs.
to_reassign = {"valid": set(), "test": set()}

for row in rows:
    if row["dataset_a"] != "v1" or row["dataset_b"] != "v1":
        continue
    if row["split_a"] != "train" or row["split_b"] not in ("valid", "test"):
        continue
    to_reassign[row["split_b"]].add(row["file_b"])

expected = {"valid": 5, "test": 4}
for split, count in expected.items():
    if len(to_reassign[split]) != count:
        raise SystemExit(
            f"STOP: Expected {count} unique {split} files, "
            f"found {len(to_reassign[split])}."
        )

# Verify all source image/label pairs before creating the new dataset.
for split, filenames in to_reassign.items():
    folder_split = "val" if split == "valid" else split
    for filename in filenames:
        image_path = source / "images" / folder_split / filename
        label_path = source / "labels" / folder_split / (Path(filename).stem + ".txt")
        if not image_path.is_file() or not label_path.is_file():
            raise SystemExit(f"STOP: Missing image or label: {filename}")

# Copy only dataset content. Original dataset remains untouched.
destination.mkdir(parents=True)
for kind in ("images", "labels"):
    for split in ("train", "val", "test"):
        src = source / kind / split
        dst = destination / kind / split
        dst.mkdir(parents=True)
        for file in src.iterdir():
            if file.is_file():
                shutil.copy2(file, dst / file.name)

# Reassign validation/test near-duplicates into train.
actions = []
for split, filenames in to_reassign.items():
    folder_split = "val" if split == "valid" else split
    for filename in sorted(filenames):
        image_name = filename
        label_name = Path(filename).stem + ".txt"

        image_src = destination / "images" / folder_split / image_name
        label_src = destination / "labels" / folder_split / label_name
        image_dst = destination / "images" / "train" / image_name
        label_dst = destination / "labels" / "train" / label_name

        if image_dst.exists() or label_dst.exists():
            raise SystemExit(f"STOP: Filename collision: {filename}")

        shutil.move(str(image_src), str(image_dst))
        shutil.move(str(label_src), str(label_dst))
        actions.append((split, filename, "train"))

# Create YAML for the corrected dataset.
(destination / "data.yaml").write_text(
    "path: .\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n\n"
    "names:\n"
    "  0: fire\n"
    "  1: smoke\n",
    encoding="utf-8"
)

# Verify every image has a label and summarize polygons/classes.
summary = {}
errors = []

for split in ("train", "val", "test"):
    image_dir = destination / "images" / split
    label_dir = destination / "labels" / split
    images = [p for p in image_dir.iterdir() if p.is_file()]
    labels = {p.name for p in label_dir.glob("*.txt")}

    missing = [p.name for p in images if p.stem + ".txt" not in labels]
    if missing:
        errors.extend([f"{split}: missing label for {x}" for x in missing])

    polygon_count = 0
    fire_count = 0
    smoke_count = 0
    empty_count = 0

    for label in label_dir.glob("*.txt"):
        lines = [x.strip() for x in label.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            empty_count += 1
        for line in lines:
            class_id = line.split()[0]
            polygon_count += 1
            if class_id == "0":
                fire_count += 1
            elif class_id == "1":
                smoke_count += 1
            else:
                errors.append(f"{split}: unknown class {class_id} in {label.name}")

    summary[split] = {
        "images": len(images),
        "labels": len(labels),
        "polygons": polygon_count,
        "fire": fire_count,
        "smoke": smoke_count,
        "empty_labels": empty_count
    }

if errors:
    print("\nVERIFICATION ERRORS:")
    for error in errors:
        print(error)
    raise SystemExit("Dataset copy created, but verification needs attention.")

report = [
    "FIRESIGHT AI - SPLIT-SAFE DATASET REPORT",
    "Original dataset preserved unchanged.",
    "Visual near-duplicate counterparts reassigned to train.",
    "",
]
for split, stats in summary.items():
    report.append(
        f"{split}: images={stats['images']}, labels={stats['labels']}, "
        f"polygons={stats['polygons']}, fire={stats['fire']}, "
        f"smoke={stats['smoke']}, empty_labels={stats['empty_labels']}"
    )

report += [
    "",
    f"Reassigned from validation: {len(to_reassign['valid'])}",
    f"Reassigned from test: {len(to_reassign['test'])}",
    f"Total reassigned: {len(actions)}",
    "Verification errors: 0",
]

report_path = Path("reports/results/split_safe_dataset_report.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(report), encoding="utf-8")

action_path = Path("reports/results/split_safe_reassignments.csv")
with action_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["original_split", "image", "new_split"])
    writer.writerows(actions)

print("\nSPLIT-SAFE DATASET CREATED AND VERIFIED")
print("Location:", destination.resolve())
for split, stats in summary.items():
    print(
        f"{split}: images={stats['images']}, labels={stats['labels']}, "
        f"polygons={stats['polygons']}, fire={stats['fire']}, "
        f"smoke={stats['smoke']}, empty={stats['empty_labels']}"
    )

print("\nReassigned validation:", len(to_reassign["valid"]))
print("Reassigned test:", len(to_reassign["test"]))
print("Verification errors:", len(errors))
print("Report:", report_path)
print("Action log:", action_path)
