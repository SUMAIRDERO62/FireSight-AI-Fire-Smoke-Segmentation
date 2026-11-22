# FireSight AI — Dataset Selection
**Phase:** 01 — Selection decision  
**Research date:** 21 November 2026  
**Status:** Proposed selection; awaiting project owner approval.

## 1. Decision summary

**Do not approve a single existing Kaggle dataset as the final segmentation ground truth yet.**

The reviewed Kaggle sources are either:
- verified detection-only (D-Fire);
- insufficiently documented at file level (DataCluster Labs listing); or
- described as YOLO bounding-box detection data (Fire and Smoke YOLO dataset).

The strongest segmentation-format candidate found during research is a non-Kaggle Hugging Face dataset with 1,348 images and fire polygons, but its annotations are machine-generated, its class scope is fire-only, and its upstream image rights require further checking. It is not an adequate standalone source for FireSight AI's fire + smoke segmentation goal.

## 2. Recommended data strategy

Adopt a **verified multi-source annotation strategy**, with a small manually reviewed segmentation benchmark and larger supplementary detection/negative pools.

### Proposed roles

| Role | Candidate | Use | Approval |
|---|---|---|---|
| Supplementary fire/smoke and negatives | D-Fire (Kaggle) | Detection pretraining reference, hard negatives, image diversity; obtain segmentation labels separately | Conditional on source-license review |
| Segmentation-format pilot | sreeharivp23 fire segmentation (Hugging Face) | Fire-only pipeline smoke test / annotation-format pilot; not final ground truth | Conditional; machine labels must be audited |
| Additional dual-class image pool | DataCluster Labs Kaggle item | Consider only after inspecting archive, annotation schema, and license | Pending |
| Additional dual-class detection pool | azimjaan21 Fire and Smoke YOLO (Kaggle) | Supplementary images only; new polygons required | Pending archive/license verification |
| Final ground-truth benchmark | Manually annotated, representative fire + smoke set | Trusted U-Net and YOLO11-seg validation/test | Required before claiming segmentation quality |

## 3. Why this is the safest technical choice

### U-Net needs pixel-level ground truth

For a semantic segmentation model, the target is a pixel map. A rectangular region generated from a bounding box teaches the network that the entire box is fire or smoke, including background pixels inside the rectangle. This is especially damaging for smoke, whose boundaries are diffuse and irregular.

### YOLO11-seg needs instance polygons

YOLO11-seg requires polygon vertices per object instance. A detection box does not contain enough information to reconstruct the actual contour. Converting a box into a four-corner polygon merely trains the model to segment rectangles.

### A unified polygon source can serve both models

The preferred canonical annotation should be instance polygons for both classes:
- `fire`
- `smoke`

From those annotations:
- keep polygons for YOLO11-seg;
- rasterize them into indexed semantic masks for U-Net;
- store empty labels/all-background masks for negative images;
- define a consistent rule for overlapping fire/smoke pixels.

This avoids maintaining two independently annotated datasets and supports a fair model comparison.

## 4. Proposed dataset acceptance gate

A candidate must pass every required gate before approval.

| Gate | Required evidence | Pass condition |
|---|---|---|
| Identity | Official Kaggle URL, owner, version | Source and version recorded |
| Images | Actual archive inventory | Count, extensions, dimensions, corrupt files known |
| Classes | File-level labels and visual sample | Fire, smoke, both, and negative counts measured |
| Segmentation | Actual masks or polygon annotations | Pixel masks or per-instance polygons exist; boxes alone fail |
| Provenance | Annotation documentation | Human vs machine labels identified; generation method recorded |
| Quality | Visual audit | Representative masks align with visible boundaries |
| Splits | Official split files or reproducible split plan | Train/validation/test separated by source/video/scene where possible |
| Leakage | Hash and near-duplicate audit | No duplicate or adjacent video frames across splits |
| License | License text and upstream source terms | Research/training use permitted; attribution obligations known |
| Model compatibility | Conversion test on sample records | Both U-Net mask and YOLO segmentation label can be generated without guessing |

**Mandatory rejection conditions**
- Only image-level classification labels.
- Only bounding boxes, when presented as segmentation ground truth.
- Unknown or conflicting rights for intended use.
- No way to identify negative samples reliably.
- Severe label errors or untraceable machine-generated annotations.

## 5. Proposed class and mask convention

| Class ID | Name | U-Net mask value |
|---:|---|---:|
| 0 | Background | 0 |
| 1 | Fire | 1 |
| 2 | Smoke | 2 |

For YOLO11-seg, the class IDs should match the same class order:

```yaml
names:
  0: fire
  1: smoke
```

The U-Net mask uses a separate background value `0`, so its fire/smoke pixel values are shifted by one relative to the YOLO class IDs. This mapping must be explicit in the preparation code.

If fire and smoke polygons overlap, define and document a deterministic overlap policy before rasterization. If instance identity and overlap are important, preserve the original polygons and instance metadata even after generating semantic masks.

## 6. Split recommendation

Do not blindly combine pre-existing split folders and then randomly resplit the merged images. This can leak near-identical images or adjacent video frames between training and evaluation.

Recommended approach after dataset approval:
1. Preserve official splits where their provenance is reliable.
2. Group images by original source, video, camera, event, or scene when metadata permits.
3. Allocate groups—not individual near-duplicate frames—to train, validation, and test.
4. Use validation for model selection.
5. Keep test data untouched until final evaluation.
6. Use the same image IDs and split assignments for U-Net and YOLO11-seg.

No split percentages are finalized in this research phase.

## 7. Annotation quality plan

Before training:
- Manually inspect a stratified sample of fire, smoke, fire+smoke, and negative images.
- Review flame tips, reflections, glow, smoke edges, haze, clouds, fog, steam, and bright lights.
- Check that negative images have empty segmentation labels and all-background masks.
- Audit machine-generated masks against human review; do not use auto-label validation scores as independent ground truth.
- Record annotation provenance at image level where mixed manual/automatic labels are used.
- Prefer manual correction of a smaller, high-quality validation/test set over relying on noisy labels.

## 8. Final recommendation

**Recommended decision: conditional approval of D-Fire as supplementary data only; no final segmentation dataset approved yet.**

Do not use D-Fire's YOLO bounding boxes as segmentation masks. Keep the Hugging Face fire-polygon dataset as an optional experimental pilot, not the final two-class dataset. Hold the DataCluster Labs and azimjaan21 Kaggle candidates until their archives and licenses can be verified.

For a credible FireSight AI comparison, the project needs a dual-class fire-and-smoke instance-polygon dataset, or a manually annotated subset created from licensed source images. One well-audited polygon dataset can support both U-Net and YOLO11-seg.

## 9. Approval required before Phase 02

Please approve one of these directions:

- **Option A — Recommended:** Proceed to a limited verification/annotation feasibility stage. Inspect the candidate archives and licenses (without training), then decide whether to annotate a high-quality dual-class subset.
- **Option B:** Use D-Fire as the main image pool and manually create fire/smoke polygons for the selected images. This offers image scale but requires substantial annotation work.
- **Option C:** Pilot the Hugging Face fire-only polygon dataset for pipeline-format testing, while keeping smoke segmentation and final ground truth on hold.

No large dataset downloads, preprocessing pipeline, mask generation at scale, or model training should begin until you explicitly approve the selected direction.

## 10. Source links

- D-Fire Kaggle: https://www.kaggle.com/datasets/shubhamkarande13/d-fire
- D-Fire original repository: https://github.com/gaia-solutions-on-demand/DFireDataset
- D-Fire license: https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/LICENSE
- DataCluster Labs Kaggle: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
- azimjaan21 Kaggle YOLO dataset: https://www.kaggle.com/datasets/azimjaan21/fire-and-smoke-dataset-object-detection-yolo
- Hugging Face fire polygons: https://huggingface.co/datasets/sreeharivp23/fire-segmentation-dataset
- DBSeg research lead: https://ayazkhan789.github.io/DBSeg/
