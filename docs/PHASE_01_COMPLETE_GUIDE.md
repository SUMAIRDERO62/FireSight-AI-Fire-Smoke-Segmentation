# FireSight AI — Phase 01 Complete Kit

## What this phase does
Phase 01 establishes what is actually present in the downloaded D-Fire and DataCluster folders. It inventories files, counts image and likely annotation formats, reports directory-based split hints, checks approximate image/annotation filename-stem matching, and inspects representative TXT/JSON/XML annotations.

It does not train models, create masks, convert labels, or alter original datasets.

## Your existing project
Your terminal log showed an established architecture with `app/`, `configs/`, `data/`, `docs/`, `models/`, `notebooks/`, `reports/`, `scripts/`, `src/`, and `tests/`. This kit adds files to the existing `scripts/` and `docs/` folders; it does not replace the architecture.

Expected raw dataset paths:
- `data/raw/D-Fire`
- `data/raw/DataCluster`

The DataCluster folder should contain its extracted `Annotations/` and `Datacluster Fire and Smoke Sample/` directories if those are how your download was extracted.

## Install the kit
Extract the ZIP contents into `E:\BEST\FireSight AI`. Merge folders. Do not overwrite unrelated files.

## Run with PowerShell
Option A — create any missing Phase 01 folders:
```powershell
cd "E:\BEST\FireSight AI"
powershell -ExecutionPolicy Bypass -File ".\scripts\create_phase01_structure.ps1"
```

Option B — run the audit:
```powershell
cd "E:\BEST\FireSight AI"
powershell -ExecutionPolicy Bypass -File ".\scripts\run_phase01.ps1"
```

You can also run Python directly:
```powershell
python .\scripts\inspect_datasets.py
```

The audit uses Python standard library only. No pip installation is required.

## Output reports
Created under `data/inspection/phase01/`:
- `phase01_dataset_audit.txt`
- `phase01_dataset_audit.json`
- `phase01_dataset_summary.csv`

Upload TXT and JSON here. CSV is optional.

## How to interpret
- YOLO rows with 5 numbers (class + x_center + y_center + width + height) are box-shaped detection labels, not segmentation polygons.
- YOLO rows with class plus 3 or more coordinate pairs may be polygon-shaped segmentation labels. The script flags the row shape; class semantics and coordinate validity still require review.
- COCO annotation objects may contain `bbox` and `segmentation`. The report counts polygon lists, RLE dictionaries and empty/missing segmentation.
- Pascal VOC XML with `bndbox` indicates boxes; polygon/mask tags need manual inspection.
- Matching filename stems is approximate. It is not proof that every image has the correct label.
- Split names are hints based on folder names, not proof that splits are leakage-free.
- TFRecord files are counted but not decoded.
- The script does not determine license or upstream image rights.

## Important model requirements
U-Net needs pixel-level semantic masks (for example background=0, fire=1, smoke=2).
YOLO11-seg needs per-instance polygons/masks.
Bounding boxes alone cannot be converted into ground-truth segmentation masks.
If labels are detection-only, classify that dataset as detection-only or plan a separately approved annotation effort. Do not invent masks.

## Git and data safety
Do not commit raw datasets or large model weights to GitHub. Review `.gitignore` and ensure `data/raw/` and generated data/processed outputs are excluded as appropriate. A suggested snippet is in `configs/phase01_gitignore_snippet.txt`; merge it manually with the existing `.gitignore` rather than replacing the file.
