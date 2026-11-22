from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_yolo(path: Path):
    rows, errors, classes = 0, [], Counter()
    if not path.exists():
        return rows, ["missing_label_file"], classes
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        return rows, [f"read_error:{exc}"], classes
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) != 5:
            errors.append(f"line_{line_no}:expected_5_fields_got_{len(parts)}")
            continue
        try:
            cid = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            if cid < 0:
                raise ValueError("negative_class_id")
            if not all(0.0 <= x <= 1.0 for x in vals):
                raise ValueError("normalized_coordinate_out_of_range")
            if vals[2] <= 0 or vals[3] <= 0:
                raise ValueError("nonpositive_box_size")
            classes[str(cid)] += 1
            rows += 1
        except Exception as exc:
            errors.append(f"line_{line_no}:{exc}")
    return rows, errors, classes

def parse_voc(path: Path):
    result = {"objects": 0, "classes": Counter(), "segmented": None, "errors": []}
    try:
        root = ET.parse(path).getroot()
        seg = root.findtext("segmented")
        result["segmented"] = seg
        size = root.find("size")
        width = int(float(size.findtext("width"))) if size is not None else 0
        height = int(float(size.findtext("height"))) if size is not None else 0
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "unknown").strip().lower()
            box = obj.find("bndbox")
            if box is None:
                result["errors"].append("object_missing_bndbox")
                continue
            coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            xmin, ymin, xmax, ymax = coords
            if width and height and not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                result["errors"].append("bbox_out_of_image_bounds")
            result["classes"][name] += 1
            result["objects"] += 1
    except Exception as exc:
        result["errors"].append(f"xml_error:{exc}")
    return result

def discover_dataset(root: Path, name: str):
    records, summary = [], {"name": name, "root": str(root), "image_count": 0,
                            "annotation_count": 0, "matched_count": 0,
                            "unmatched_images": 0, "orphan_annotations": 0,
                            "annotation_formats": Counter(), "classes": Counter(),
                            "errors": Counter(), "dimensions": Counter(),
                            "segmentation_evidence": Counter()}
    if not root.exists():
        summary["error"] = "dataset_root_not_found"
        return records, summary

    image_map, label_map = {}, {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            image_map.setdefault(p.stem.lower(), []).append(p)
        elif p.suffix.lower() in {".txt", ".xml"}:
            label_map.setdefault(p.stem.lower(), []).append(p)

    summary["image_count"] = sum(map(len, image_map.values()))
    summary["annotation_count"] = sum(map(len, label_map.values()))
    all_hashes = defaultdict(list)

    for stem, imgs in image_map.items():
        for img in imgs:
            ann_candidates = label_map.get(stem, [])
            # Prefer labels in a sibling labels folder or same relative split.
            ann = None
            for candidate in ann_candidates:
                if candidate.suffix.lower() == ".txt" and "labels" in candidate.parts:
                    ann = candidate; break
                if candidate.suffix.lower() == ".xml":
                    ann = candidate; break
            row = {"dataset": name, "stem": stem, "image": str(img), "annotation": str(ann) if ann else "",
                   "split_source": "train" if "train" in [x.lower() for x in img.parts] else
                                   "test" if "test" in [x.lower() for x in img.parts] else "unassigned",
                   "image_width": "", "image_height": "", "sha256": "", "annotation_format": "none",
                   "object_count": 0, "classes": "", "valid": True, "errors": ""}
            try:
                from PIL import Image
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    row["image_width"], row["image_height"] = im.size
                    summary["dimensions"][f"{im.width}x{im.height}"] += 1
            except ImportError:
                row["errors"] = "Pillow_not_installed_dimensions_not_checked"
            except Exception as exc:
                row["valid"] = False
                row["errors"] = f"image_error:{exc}"
            try:
                digest = sha256(img)
                row["sha256"] = digest
                all_hashes[digest].append(str(img))
            except Exception as exc:
                row["valid"] = False
                row["errors"] = (row["errors"] + ";" if row["errors"] else "") + f"hash_error:{exc}"

            if ann:
                if ann.suffix.lower() == ".txt":
                    row["annotation_format"] = "yolo_detection_candidate"
                    n, errs, cls = parse_yolo(ann)
                    row["object_count"] = n
                    row["classes"] = ";".join(f"{k}:{v}" for k,v in sorted(cls.items()))
                    summary["classes"].update(cls)
                    summary["annotation_formats"]["yolo_txt"] += 1
                    if errs:
                        row["valid"] = False
                        row["errors"] = (row["errors"] + ";" if row["errors"] else "") + ";".join(errs)
                    summary["segmentation_evidence"]["yolo_polygon_rows"] += 0
                else:
                    row["annotation_format"] = "pascal_voc_detection_candidate"
                    voc = parse_voc(ann)
                    row["object_count"] = voc["objects"]
                    row["classes"] = ";".join(f"{k}:{v}" for k,v in sorted(voc["classes"].items()))
                    summary["classes"].update(voc["classes"])
                    summary["annotation_formats"]["pascal_voc_xml"] += 1
                    summary["segmentation_evidence"][f"segmented_tag_{voc['segmented']}"] += 1
                    if voc["errors"]:
                        row["valid"] = False
                        row["errors"] = (row["errors"] + ";" if row["errors"] else "") + ";".join(voc["errors"])
                summary["matched_count"] += 1
            else:
                summary["unmatched_images"] += 1
                row["errors"] = (row["errors"] + ";" if row["errors"] else "") + "missing_annotation"
            if not row["valid"]:
                summary["errors"]["invalid_records"] += 1
            records.append(row)

    summary["orphan_annotations"] = sum(1 for stem in label_map if stem not in image_map)
    summary["classes"] = dict(summary["classes"])
    summary["annotation_formats"] = dict(summary["annotation_formats"])
    summary["dimensions"] = dict(summary["dimensions"])
    summary["segmentation_evidence"] = dict(summary["segmentation_evidence"])
    summary["errors"] = dict(summary["errors"])
    duplicates = [{"sha256": h, "count": len(paths), "paths": paths}
                  for h, paths in all_hashes.items() if len(paths) > 1]
    summary["exact_duplicate_groups"] = len(duplicates)
    summary["exact_duplicate_images_in_groups"] = sum(x["count"] for x in duplicates)
    return records, summary, duplicates

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    project = Path(args.project_root).resolve()
    out = project / "reports" / "phase02"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    specs = [("D-Fire", project / "data/raw/D-Fire"),
             ("DataCluster", project / "data/raw/DataCluster")]
    all_rows, summaries, dup_rows = [], [], []
    for name, root in specs:
        result = discover_dataset(root, name)
        if len(result) == 3:
            rows, summary, dups = result
        else:
            rows, summary = result; dups = []
        all_rows.extend(rows); summaries.append(summary)
        for d in dups:
            dup_rows.append({"dataset": name, **d})

    fields = ["dataset","stem","image","annotation","split_source","image_width","image_height",
              "sha256","annotation_format","object_count","classes","valid","errors"]
    with (out/"dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    with (out/"duplicate_groups.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["dataset","sha256","count","paths"]); w.writeheader(); w.writerows(dup_rows)

    class_counts = Counter()
    for s in summaries:
        class_counts.update(s.get("classes", {}))
    with (out/"annotation_class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["class_id_or_name","object_count"])
        w.writerows(sorted(class_counts.items()))

    payload = {"phase": 2, "seed_default": 42, "datasets": summaries,
               "segmentation_readiness": "NOT_VERIFIED: inspected annotations are bounding boxes, not masks/polygons",
               "warning": "Do not synthesize segmentation masks from bounding boxes."}
    (out/"dataset_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# FireSight AI — Phase 02 Dataset Audit", "",
             "> Generated by `scripts/phase02_audit.py`. Re-run after any data changes.", "",
             "## Segmentation readiness", "",
             "**Not verified / not ready from current annotations.** D-Fire samples are 5-field YOLO bounding boxes. The inspected DataCluster XML uses Pascal VOC `bndbox` and `segmented=0`. No masks or polygons are generated.", ""]
    for s in summaries:
        lines += [f"## {s['name']}", "", f"- Image files: {s.get('image_count',0)}",
                  f"- Annotation files: {s.get('annotation_count',0)}",
                  f"- Matched image records: {s.get('matched_count',0)}",
                  f"- Images without matching annotation: {s.get('unmatched_images',0)}",
                  f"- Orphan annotations: {s.get('orphan_annotations',0)}",
                  f"- Exact duplicate groups: {s.get('exact_duplicate_groups',0)}",
                  f"- Annotation formats: `{json.dumps(s.get('annotation_formats',{}))}`",
                  f"- Class counts: `{json.dumps(s.get('classes',{}))}`",
                  f"- Dimensions: `{json.dumps(s.get('dimensions',{}))}`", ""]
    lines += ["## Interpretation", "",
              "- D-Fire is detection-labelled based on the inspected YOLO rows; class IDs remain numeric until the official class mapping is confirmed.",
              "- DataCluster is VOC detection-labelled based on the inspected XML; unmatched images must not automatically be treated as negatives.",
              "- Exact SHA-256 duplicate detection is included. Near-duplicate/video-sequence leakage requires a separate review of source metadata or perceptual grouping.",
              "- No source files under `data/raw/` are modified.", ""]
    (out/"dataset_report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        for s in summaries:
            dims=s.get("dimensions",{})
            if dims:
                plt.figure(figsize=(10,4))
                plt.bar(list(dims.keys()), list(dims.values()))
                plt.xticks(rotation=45, ha="right"); plt.title(f"{s['name']} image dimensions")
                plt.tight_layout(); plt.savefig(out/"figures"/f"{s['name'].lower()}_dimensions.png"); plt.close()
            cls=s.get("classes",{})
            if cls:
                plt.figure(figsize=(6,4))
                plt.bar(list(cls.keys()), list(cls.values()))
                plt.title(f"{s['name']} annotation class counts")
                plt.tight_layout(); plt.savefig(out/"figures"/f"{s['name'].lower()}_class_counts.png"); plt.close()
    except ImportError:
        pass
    print(f"Audit complete: {out}")
    print("Segmentation readiness: NOT VERIFIED — bounding boxes are not masks/polygons.")

if __name__ == "__main__":
    main()
