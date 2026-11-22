"""
FireSight AI | Phase 01 Dataset Audit
Read-only audit for D-Fire and DataCluster.

- Does not move, rename, edit, or delete source dataset files.
- Uses Python standard library only.
- Writes TXT, JSON and CSV reports under data/inspection/phase01/.
- Optional image dimensions are read with Pillow if installed.
"""

from __future__ import annotations
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "inspection" / "phase01"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANN_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml", ".tfrecord", ".record"}
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
SPLIT_WORDS = {"train", "training", "val", "valid", "validation", "test", "testing"}


def files_under(folder: Path):
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and not any(part in IGNORED_PARTS for part in p.parts)
    ]


def rel(path: Path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def human_size(n):
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024


def optional_image_dimensions(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return {"width": im.width, "height": im.height, "mode": im.mode}
    except ImportError:
        return {"note": "Pillow not installed; dimensions skipped"}
    except Exception as exc:
        return {"error": str(exc)}


def inspect_yolo_txt(path):
    result = {
        "format": "TXT / possible YOLO",
        "nonempty_lines": 0,
        "five_value_box_rows": 0,
        "polygon_shaped_rows": 0,
        "other_rows": 0,
        "class_ids_seen": [],
        "sample_lines": [],
        "coordinate_values_in_0_1": None,
    }
    classes = set()
    all_coords_in_range = True
    try:
        lines = [
            line.strip() for line in path.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines() if line.strip()
        ]
        result["nonempty_lines"] = len(lines)
        result["sample_lines"] = lines[:5]
        for line in lines[:10000]:
            parts = re.split(r"\s+", line)
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                result["other_rows"] += 1
                continue
            if not nums:
                result["other_rows"] += 1
                continue
            classes.add(str(int(nums[0])) if nums[0].is_integer() else str(nums[0]))
            coords = nums[1:]
            if any(x < 0 or x > 1 for x in coords):
                all_coords_in_range = False
            if len(nums) == 5:
                result["five_value_box_rows"] += 1
            elif len(nums) >= 7 and len(coords) >= 6 and len(coords) % 2 == 0:
                result["polygon_shaped_rows"] += 1
            else:
                result["other_rows"] += 1
        result["class_ids_seen"] = sorted(classes)
        result["coordinate_values_in_0_1"] = all_coords_in_range if result["nonempty_lines"] else None
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inspect_coco_json(path):
    result = {
        "format": "JSON / possible COCO",
        "top_level_keys": [],
        "images_count": None,
        "annotations_count": None,
        "categories": [],
        "bbox_annotations": 0,
        "polygon_segmentation_annotations": 0,
        "rle_segmentation_annotations": 0,
        "empty_or_missing_segmentation_annotations": 0,
        "sample_image_names": [],
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            result["note"] = "Top-level JSON is not an object."
            result["top_level_type"] = type(data).__name__
            return result
        result["top_level_keys"] = list(data.keys())[:80]
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        result["images_count"] = len(images) if isinstance(images, list) else None
        result["annotations_count"] = len(annotations) if isinstance(annotations, list) else None
        if isinstance(images, list):
            result["sample_image_names"] = [
                im.get("file_name") for im in images[:10] if isinstance(im, dict)
            ]
        if isinstance(categories, list):
            result["categories"] = [
                {"id": c.get("id"), "name": c.get("name")}
                for c in categories[:100] if isinstance(c, dict)
            ]
        if isinstance(annotations, list):
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                if ann.get("bbox"):
                    result["bbox_annotations"] += 1
                seg = ann.get("segmentation")
                if isinstance(seg, list) and len(seg) > 0:
                    result["polygon_segmentation_annotations"] += 1
                elif isinstance(seg, dict) and (seg.get("counts") is not None):
                    result["rle_segmentation_annotations"] += 1
                else:
                    result["empty_or_missing_segmentation_annotations"] += 1
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inspect_xml(path):
    result = {
        "format": "XML / possible Pascal VOC",
        "bndbox_found": False,
        "polygon_or_point_tags_found": False,
        "mask_or_segmentation_tags_found": False,
        "tags": [],
    }
    try:
        root = ET.parse(path).getroot()
        tags = [node.tag.lower() for node in root.iter()]
        result["tags"] = sorted(set(tags))[:100]
        result["bndbox_found"] = "bndbox" in tags
        result["polygon_or_point_tags_found"] = any(
            "polygon" in tag or "point" in tag for tag in tags
        )
        result["mask_or_segmentation_tags_found"] = any(
            "mask" in tag or "segmentation" in tag for tag in tags
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inspect_annotation(path):
    ext = path.suffix.lower()
    if ext == ".txt":
        return inspect_yolo_txt(path)
    if ext == ".json":
        return inspect_coco_json(path)
    if ext == ".xml":
        return inspect_xml(path)
    return {
        "format": ext or "no extension",
        "note": "File counted; this extension is not deeply parsed by this script."
    }


def split_counts(files, dataset_root):
    counts = Counter()
    for p in files:
        try:
            parts = [part.lower() for part in p.relative_to(dataset_root).parts[:-1]]
        except ValueError:
            continue
        found = next((part for part in parts if part in SPLIT_WORDS), None)
        counts[found or "(split not identified)"] += 1
    return dict(counts)


def match_stems(images, annotations):
    image_stems = defaultdict(list)
    ann_stems = defaultdict(list)
    for p in images:
        image_stems[p.stem.lower()].append(p)
    for p in annotations:
        if p.suffix.lower() in {".txt", ".xml", ".json"}:
            ann_stems[p.stem.lower()].append(p)
    matched = sorted(set(image_stems) & set(ann_stems))
    images_without = sorted(set(image_stems) - set(ann_stems))
    ann_without = sorted(set(ann_stems) - set(image_stems))
    return {
        "unique_image_stems": len(image_stems),
        "unique_annotation_stems_txt_xml_json": len(ann_stems),
        "matched_stems": len(matched),
        "image_stems_without_same_stem_annotation": len(images_without),
        "annotation_stems_without_same_stem_image": len(ann_without),
        "sample_unmatched_image_stems": images_without[:20],
        "sample_unmatched_annotation_stems": ann_without[:20],
        "warning": "Stem matching is approximate; nested folders and shared names may need dataset-specific checks."
    }


def audit_dataset(name):
    folder = RAW / name
    result = {
        "dataset": name,
        "path": str(folder),
        "exists": folder.exists(),
    }
    if not folder.exists():
        result["error"] = "Expected dataset folder not found."
        return result, [f"{name}: expected folder not found: {folder}"]

    all_files = files_under(folder)
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS]
    anns = [p for p in all_files if p.suffix.lower() in ANN_EXTS]
    extension_counts = Counter(p.suffix.lower() or "[no extension]" for p in all_files)
    total_bytes = sum(p.stat().st_size for p in all_files)

    result.update({
        "total_files": len(all_files),
        "total_size_bytes": total_bytes,
        "total_size_human": human_size(total_bytes),
        "image_count": len(images),
        "potential_annotation_file_count": len(anns),
        "extension_counts": dict(extension_counts),
        "split_file_counts": split_counts(all_files, folder),
        "image_annotation_stem_match": match_stems(images, anns),
        "sample_images": [
            {"path": rel(p), "size_bytes": p.stat().st_size,
             "dimensions": optional_image_dimensions(p)}
            for p in images[:8]
        ],
        "annotation_samples": [
            {"path": rel(p), "extension": p.suffix.lower(),
             "size_bytes": p.stat().st_size, "inspection": inspect_annotation(p)}
            for p in anns[:20]
        ],
    })

    lines = [
        f"DATASET: {name}",
        f"Folder exists: YES",
        f"Total files: {len(all_files)}",
        f"Total size: {human_size(total_bytes)}",
        f"Images: {len(images)}",
        f"Potential annotation files: {len(anns)}",
        "Extensions:"
    ]
    lines += [f"  {ext}: {count}" for ext, count in extension_counts.most_common()]
    lines.append("Split directory counts (file counts, not verified split assignments):")
    lines += [f"  {split}: {count}" for split, count in result["split_file_counts"].items()]
    lines += ["Image/annotation same-stem check:", json.dumps(result["image_annotation_stem_match"], indent=2)]
    lines.append("Annotation sample inspection:")
    for sample in result["annotation_samples"]:
        lines.append(f"\n  FILE: {sample['path']}")
        lines.append(json.dumps(sample["inspection"], indent=2, ensure_ascii=False)[:2500])
    lines.append("")
    return result, lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().isoformat(timespec="seconds")
    summaries, report_lines = [], [
        "FIRESIGHT AI — PHASE 01 DATASET AUDIT",
        f"Generated: {generated}",
        f"Project root: {ROOT}",
        "",
        "READ-ONLY: source files are not moved, renamed, edited, or deleted.",
        "This is an initial technical audit, not proof of annotation quality or legal permission.",
        "YOLO five-value rows usually indicate detection boxes; polygon-shaped rows may indicate segmentation.",
        "COCO segmentation fields and image-label stem matching still require manual validation.",
        "TFRecord content is counted but not decoded.",
        ""
    ]

    for name in ("D-Fire", "DataCluster"):
        summary, lines = audit_dataset(name)
        summaries.append(summary)
        report_lines += ["=" * 80] + lines

    payload = {
        "generated_at": generated,
        "project_root": str(ROOT),
        "read_only": True,
        "dataset_audits": summaries,
    }
    (OUT / "phase01_dataset_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "phase01_dataset_audit.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    with (OUT / "phase01_dataset_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "exists", "total_files", "total_size_bytes",
            "image_count", "potential_annotation_file_count"
        ])
        writer.writeheader()
        for item in summaries:
            writer.writerow({
                "dataset": item.get("dataset"),
                "exists": item.get("exists"),
                "total_files": item.get("total_files", 0),
                "total_size_bytes": item.get("total_size_bytes", 0),
                "image_count": item.get("image_count", 0),
                "potential_annotation_file_count": item.get("potential_annotation_file_count", 0),
            })

    print("\n=== FIRESIGHT AI PHASE 01 AUDIT FINISHED ===")
    print(f"Reports: {OUT}")
    print("1. phase01_dataset_audit.txt")
    print("2. phase01_dataset_audit.json")
    print("3. phase01_dataset_summary.csv")
    print("\nUpload the TXT and JSON reports in ChatGPT for annotation review.")
    print("No source dataset files were modified.")


if __name__ == "__main__":
    main()
