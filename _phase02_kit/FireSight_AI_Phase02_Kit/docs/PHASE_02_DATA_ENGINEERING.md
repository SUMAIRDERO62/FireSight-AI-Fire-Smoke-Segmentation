# Phase 02 — Dataset Engineering & Preparation

## Scope

This phase audits and prepares data without model training. The source data is treated as immutable. Outputs are written under `reports/phase02/` and `data/processed/phase02/`.

## Current annotation decision

| Dataset | Observed format | Supported task now | Segmentation status |
|---|---|---|---|
| D-Fire | YOLO 5-field normalized boxes | Object detection | Not segmentation-ready |
| DataCluster | Pascal VOC XML `bndbox`, `segmented=0` in inspected sample | Object detection after a separately reviewed conversion | Not segmentation-ready |

D-Fire class IDs have counts from the user's complete non-empty label scan: class 0 = 11,865 and class 1 = 14,692. Do not attach class names until verified from authoritative dataset documentation.

## Splitting policy

- D-Fire's original `test` partition remains test.
- Validation is carved from original `train` using a deterministic hash of image SHA-256 and seed.
- Exact duplicate hashes appearing across original train/test are excluded from the train side and listed in the preparation report.
- The method detects byte-identical duplicates only. It does not prove that related video frames or near-duplicates cannot cross splits.
- DataCluster remains separate because it has no identified split metadata and incomplete image/XML pairing in the initial audit.

## Segmentation gate

U-Net requires genuine pixel masks. YOLO11-seg requires polygon/mask instance annotations. Current bounding boxes do not meet either requirement. The project must obtain real masks/polygons through an approved segmentation dataset or manual annotation and quality review.

## Review checklist

- [ ] Inspect `dataset_report.md`.
- [ ] Confirm official D-Fire class-name mapping.
- [ ] Review invalid and unmatched records.
- [ ] Review duplicate groups.
- [ ] Inspect a sample of prepared images and labels.
- [ ] Confirm no near-duplicate/source-sequence leakage using source metadata.
- [ ] Do not train segmentation models until genuine masks/polygons exist.
