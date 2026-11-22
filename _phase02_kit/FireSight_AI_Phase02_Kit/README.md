# FireSight AI — Phase 02 Data Engineering Kit

This kit prepares and audits the datasets already present in your project. It does **not** train models and does **not** fabricate segmentation masks.

## Evidence-based dataset status

- **D-Fire:** YOLO detection bounding boxes (`class x_center y_center width height`). The audit found 11,865 rows with class ID `0` and 14,692 with class ID `1`. No class-name YAML was found in the inspected dataset directory, so class names are deliberately left unresolved.
- **DataCluster:** Pascal VOC XML bounding boxes. The inspected XML contains `<segmented>0</segmented>` and a `bndbox`, not polygon/mask annotations.
- Therefore, neither dataset is currently verified as ground-truth segmentation data for U-Net or YOLO11-seg.

## What this kit does

1. Audits image/annotation pairing and annotation formats.
2. Checks image readability, dimensions, hashes, malformed labels, and exact duplicate images.
3. Creates CSV/JSON reports and optional charts.
4. Prepares a **D-Fire YOLO detection** copy while retaining its original test partition and carving validation data only from the original training partition.
5. Uses a fixed seed and SHA-256-based deterministic assignment.
6. Includes tests for parsing and splitting utilities.

## What it intentionally does not do

- Does not convert bounding boxes into polygons or segmentation masks.
- Does not call class ID 0 “fire” or class ID 1 “smoke” without a verified class map.
- Does not mix DataCluster into D-Fire.
- Does not overwrite anything under `data/raw/`.
- Does not train U-Net, YOLO, or any other model.

## Install

From the project root, with your `.venv` activated:

```powershell
python -m pip install -r requirements-phase02.txt
```

## Run audit

```powershell
python scripts/phase02_audit.py --project-root .
```

Outputs are written to `reports/phase02/`:
- `dataset_audit.json`
- `dataset_manifest.csv`
- `annotation_class_counts.csv`
- `duplicate_groups.csv`
- `dataset_report.md`
- `figures/` (if matplotlib is installed)

## Prepare D-Fire detection data

First review the audit. Then run:

```powershell
python scripts/prepare_phase02.py --project-root . --dataset d-fire --val-ratio 0.15 --seed 42
```

This creates a separate detection-ready copy in:

```text
data/processed/phase02/d-fire-detection/
  images/train|val|test/
  labels/train|val|test/
  split_manifest.csv
  preparation_report.json
```

The original D-Fire train/test partition is preserved: the original test split remains test, while validation is carved from the original training split. Exact duplicate image hashes are kept in the same group; if a duplicate spans the original train/test boundary, the non-test copy is excluded from the prepared set and documented.

**Important:** This copy is suitable for YOLO object detection workflows, not YOLO11-seg. It is not a U-Net mask dataset.

## Tests

```powershell
python -m pytest tests/phase02 -q
```

## Before using prepared data

1. Review `reports/phase02/dataset_report.md`.
2. Verify class semantics from the original dataset publisher/source documentation.
3. Review all excluded records in the manifest/report.
4. Confirm that your chosen training code expects YOLO detection labels.
5. For U-Net or YOLO11-seg, obtain genuine masks/polygons or manually annotate them.

## Git

The ZIP contains a `.gitignore` snippet, not a replacement `.gitignore`. Merge it manually if needed. Do not delete your existing ignore rules.

```powershell
git status
git add scripts/phase02_audit.py scripts/prepare_phase02.py src/firesight/data tests/phase02 docs/PHASE_02_DATA_ENGINEERING.md requirements-phase02.txt configs/phase02_gitignore_snippet.txt README_PHASE02_KIT.md
git commit -m "feat: add phase 02 dataset audit and detection preparation"
```

If you copy this kit's README into your project, merge its content into your existing root README rather than blindly replacing it.
