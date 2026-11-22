# FireSight AI — Dataset Research
**Phase:** 01 — Kaggle dataset research and selection  
**Research date:** 21 November 2026  
**Project root (user workstation):** `E:\BEST\FireSight AI`  
**Status:** Research only. No datasets downloaded; no training performed.

## 1. Research objective and acceptance criteria

Identify image datasets suitable for two segmentation tasks:

- **U-Net semantic segmentation:** requires an image and a pixel-aligned semantic mask. For a three-class setup, mask pixel values should consistently represent `0=background`, `1=fire`, `2=smoke`.
- **YOLO11-seg instance segmentation:** requires object-instance polygons (or masks that can be converted into polygons) in Ultralytics segmentation label format. Each polygon must correspond to an individual object instance and class.

A dataset is not considered segmentation-ready merely because it has a fire/smoke class label or a bounding box. Classification labels and bounding boxes are not pixel-level masks. Bounding-box-to-rectangle conversion creates pseudo-labels, not ground truth.

## 2. Executive findings

The strongest verified Kaggle candidate examined, **D-Fire**, is a large fire/smoke detection dataset with YOLO bounding boxes. It is valuable supplementary data but does **not** directly satisfy either segmentation model's annotation requirement.

The Kaggle listing **Fire and Smoke Dataset** by DataCluster Labs is relevant, but the publicly accessible Kaggle page did not expose enough machine-readable details to verify its file-level annotation structure, counts, splits, or exact license. A related DataCluster Labs sample described elsewhere uses Pascal VOC bounding boxes, not masks; it is not safe to assume the Kaggle item has segmentation masks.

A small fire-only segmentation dataset is publicly documented on Hugging Face (not verified as a Kaggle dataset). It contains YOLO segmentation polygons and negative images, but its labels are machine-generated and smoke segmentation is not established. It is therefore a possible experimental source, not trusted ground truth.

**Current conclusion:** No reviewed Kaggle candidate has been verified as a complete, manually annotated, dual-class (fire + smoke) segmentation dataset. We should not begin training from the currently verified Kaggle data alone.

## 3. Candidate comparison

| Candidate | Platform / source | Published image count | Classes / distribution | Annotation type | Pixel masks? | U-Net | YOLO11-seg | Decision |
|---|---|---:|---|---|---|---|---|---|
| D-Fire | Kaggle; original GAIA repository | 21,527 | 1,164 fire-only; 5,867 smoke-only; 4,658 fire+smoke; 9,838 neither | YOLO detection boxes; 14,692 fire boxes and 11,865 smoke boxes | No | Not directly | Not directly | Supplementary detection/negative images only |
| Fire and Smoke Dataset (DataCluster Labs) | Kaggle | Not independently verified | Not independently verified | Kaggle file-level format not verified; related DataCluster sample documents VOC boxes | Not verified | Not approved | Not approved | Hold pending file/card inspection |
| Fire and Smoke Dataset Object Detection YOLO | Kaggle dataset referenced by a public project | Approximately 13,794 inferred from reported train/val counts; not confirmed from Kaggle data card | Fire and smoke | YOLO bounding boxes (`class x_center y_center width height`) | No | Only weak rectangular pseudo-labels | No, not without polygon labels | Supplementary only; exact Kaggle metadata must be rechecked |
| Fire Segmentation Dataset (`sreeharivp23`) | Hugging Face, not Kaggle | 1,348 | 1,098 fire-positive; 250 negatives; smoke class not established | YOLO segmentation polygons | Yes, polygon-derived masks | Yes, after rasterizing polygons | Yes, subject to validation | Experimental fire-only candidate; machine-generated labels |
| Fire & Smoke segmentation research lead (DBSeg project page) | Research project page; not verified as downloadable Kaggle dataset | 5,612 reported | Fire + smoke | Polygon instance annotations claimed | Claimed | Potentially | Potentially | Lead for further source/availability/licensing verification; not selected |

**Count note:** D-Fire category counts sum to 21,527. Its official repository describes 26,557 YOLO bounding boxes. A third-party paper also reports 21,527 images. The approximately 13,794 figure for the Kaggle “Fire and Smoke YOLO” item is derived from a public repository's reported ~11,035 train and ~2,759 validation images; it is not a verified official dataset-card total and must not be treated as authoritative.

## 4. Detailed candidate reviews

### 4.1 D-Fire — Kaggle

- Kaggle URL: https://www.kaggle.com/datasets/shubhamkarande13/d-fire
- Original project: https://github.com/gaia-solutions-on-demand/DFireDataset
- Official project page: https://gaia-solutions-on-demand.github.io/DFireDataset/

**Verified dataset summary**

| Category | Images |
|---|---:|
| Only fire | 1,164 |
| Only smoke | 5,867 |
| Fire and smoke | 4,658 |
| Neither fire nor smoke | 9,838 |
| **Total** | **21,527** |

| Box class | Bounding boxes |
|---|---:|
| Fire | 14,692 |
| Smoke | 11,865 |
| **Total** | **26,557** |

**Annotation format and mask status**

The original project states that annotations follow YOLO format with normalized coordinates and provides a utility to convert normalized coordinates to pixels. This is object detection annotation, not segmentation. The original documentation does not claim polygon masks or pixel-level masks. No manually annotated masks are established by the reviewed documentation.

**Splits**

The original project says training, validation, and test sets are available through its download resources. The exact split counts/ratios were not stated in the inspected README, so they are recorded as **available, exact counts not verified**. Do not assume a conventional 70/20/10 or 80/10/10 split.

**License and attribution**

The repository includes a CC0 1.0 license for the collection. However, the project explicitly warns that some images originate from third-party public sources and remain subject to their original providers' terms. It specifically names HPWREN, the Wildfire Observers and Smoke Recognition project / FESB, and FireNet. Users should check source-specific terms and credit the relevant providers, particularly for commercial use. Therefore, do not treat the repository-level CC0 statement as proof that every embedded image is unrestricted.

**Advantages**
- Large dataset with fire, smoke, co-occurrence, and negative scenes.
- Includes varied sources, including surveillance footage and public imagery.
- Has pre-existing train/validation/test resources.
- Useful for detection pretraining, negative-scene mining, and later annotation expansion.

**Limitations**
- Original annotations are bounding boxes only.
- Boxes cannot be relabelled as accurate segmentation polygons.
- Rectangular masks derived from boxes are coarse pseudo-labels and will misrepresent irregular flames and diffuse smoke.
- Mixed source provenance requires license/attribution review.

**Compatibility**
- U-Net: not directly compatible; a pixel mask must be independently annotated or carefully pseudo-labelled.
- YOLO11-seg: not directly compatible; detection boxes are not instance polygons.
- Recommended role: supplementary image pool and negative examples, subject to deduplication, license checks, and new segmentation annotation.

### 4.2 Fire and Smoke Dataset — DataCluster Labs (Kaggle)

- Kaggle URL: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
- Publisher profile: https://www.kaggle.com/dataclusterlabs/datasets

The Kaggle listing describes a dataset for early fire and smoke detection and smart-camera/fire-alarm applications. The accessible listing did not provide enough verifiable file-level information to establish exact image count, class counts, annotation files, segmentation mask availability, split structure, or the applicable license terms.

A related DataCluster Labs sample published on Hugging Face describes approximately 100–200 sample images with Pascal VOC XML bounding-box annotations. Its page states that the full dataset (advertised as 10,000+ images) is commercially licensed, and the sample has restrictions including no commercial use and no redistribution. This related sample is useful evidence that the provider's data may be detection-oriented, but it is **not proof** of the exact contents or license of the Kaggle upload.

**Decision:** Do not classify this Kaggle dataset as segmentation-ready. Before considering it, inspect the Kaggle download contents and data card for:
1. image count and class distribution;
2. presence of actual mask PNG/TIFF files or COCO segmentation polygons;
3. whether annotation objects contain polygons/RLE rather than only bounding boxes;
4. split definitions and duplicate leakage;
5. license and redistribution/training restrictions.

### 4.3 Fire and Smoke Dataset Object Detection YOLO — Kaggle reference

- Kaggle slug referenced by a public project: https://www.kaggle.com/datasets/azimjaan21/fire-and-smoke-dataset-object-detection-yolo
- Public project describing its use: https://github.com/phattien12/Smoke-Fire-Segmentation

A public project reports approximately 11,035 training images and 2,759 validation images and describes labels as YOLO detection text files with normalized `class_id x_center y_center width height` values. It creates rectangular pixel masks by filling bounding boxes. The project itself warns that these are coarse approximations because fire and smoke boundaries are irregular.

The approximately 13,794 total is a calculation from the reported split figures, not an independently verified Kaggle data-card count. No separate test set was established in the reviewed project description. The exact class counts, original label provenance, manual/automatic status, and dataset license were not verified.

**Decision:** Treat as detection-only unless direct inspection proves otherwise. Do not use rectangle-filled pseudo-masks as ground truth for final U-Net evaluation or as YOLO11-seg polygon labels.

### 4.4 Fire Segmentation Dataset — Hugging Face (non-Kaggle reference)

- Dataset: https://huggingface.co/datasets/sreeharivp23/fire-segmentation-dataset
- Duplicate/preview page: https://huggingface.co/datasets/Pravesh-21/fire-segmentation-dataset
- Related model card: https://huggingface.co/sreeharivp23/fire-segmentation-yolo11n

This is not a Kaggle candidate, but it is a relevant segmentation-format reference.

**Published structure**
- Training: 1,146 images.
- Validation: 202 images.
- Total: 1,348 images.
- Positive fire images: 1,098.
- Negative images: 250.
- One class (`fire`) is documented; smoke segmentation is not established.

**Annotation format**

Ultralytics YOLO segmentation `.txt` rows are documented in the form:

`class_id x1 y1 x2 y2 ...`

The coordinates are normalized polygon vertices. Polygon masks can be rasterized into semantic masks for U-Net. Negative images use empty label files.

**Label provenance**

The dataset card explicitly says labels are machine-generated:
1. a pretrained YOLOv8 fire/smoke detector proposed fire regions;
2. SAM 2.1 (base) generated masks from those boxes;
3. OpenCV contours and Douglas–Peucker simplification converted masks into polygons;
4. an HSV flame-colour heuristic supplied fallback detections for 219 images;
5. 12 images with no usable result were dropped.

This process can create false positives, missed flames, incorrect boundaries, and errors around smoke/glow or small/occluded flames. The published YOLO11n-seg validation metrics are evaluated against these automatically generated labels, not an independent manually annotated ground-truth test set.

**License**

The dataset card describes the source imagery as originating from CAIR Fire-Detection-Image-Dataset and DeepQuestAI Fire-Smoke-Dataset and instructs users to check upstream terms, particularly before commercial use. A clear, unified license granting unrestricted rights to every source image was not established in the reviewed material. Treat as **mixed-source / rights review required**.

**Decision:** Potential experimental fire-only segmentation data. Do not treat its masks as verified human ground truth; it does not satisfy the required two-class fire-and-smoke segmentation scope.

### 4.5 DBSeg Fire & Smoke Segmentation — research lead, not verified Kaggle data

- Project page: https://ayazkhan789.github.io/DBSeg/

The project page reports 5,612 images, fire and smoke classes, and polygon instance annotations. It also lists comparison datasets including BoWFire, FESB MLID, SMOKE5K, FSSD, FLAME, and IoT-Fire. This is a promising research lead because it describes dual-class polygon instance annotation.

However, the project page alone does not establish a Kaggle download, independently inspectable annotation files, manual annotation provenance, exact train/validation/test split, or license terms. These details must be verified from the actual release and any associated paper/repository before selection.

**Decision:** Investigate as a non-Kaggle fallback/benchmark lead. Not approved at this stage.

## 5. Annotation examples

### 5.1 YOLO detection label (not segmentation)

```text
0 0.512 0.481 0.220 0.310
1 0.630 0.390 0.300 0.420
```

These rows encode class, box centre x/y, width, and height. They do not describe the shape of the fire or smoke. They are not valid YOLO11-seg polygons.

### 5.2 YOLO11-seg polygon label

```text
0 0.41 0.22 0.46 0.18 0.53 0.24 0.55 0.39 0.49 0.51 0.42 0.44
1 0.20 0.10 0.35 0.08 0.48 0.16 0.57 0.31 0.50 0.55 0.33 0.62
```

Each row represents one object instance: class ID followed by normalized polygon vertex coordinates. The example is illustrative only, not an annotation taken from a candidate dataset.

### 5.3 U-Net semantic mask

A three-class indexed PNG mask should contain pixel values such as:

```text
0 = background
1 = fire
2 = smoke
```

This is a semantic label map, not an RGB visualization. If fire and smoke overlap, the project must define a consistent precedence/overlap policy, because a single-channel semantic mask assigns one class per pixel. An instance/panoptic representation may be needed if overlapping objects must remain distinct.

## 6. Can one unified dataset train both models?

**Yes, if the source annotation is sufficiently rich.** A dataset containing instance polygons for every fire and smoke object can support both models:

1. Keep polygons and instance IDs/classes for YOLO11-seg.
2. Rasterize polygons into a semantic mask for U-Net.
3. Resolve overlapping instances and define the background/fire/smoke pixel policy.
4. Preserve negative images with all-background masks and empty YOLO label files.
5. Use the same source split for both models to keep comparisons fair.

The reverse is not always possible. A semantic mask may merge adjacent same-class objects and lose instance boundaries. Converting it into YOLO polygons can require connected-component splitting and may not reconstruct the true object instances. Therefore, **instance polygons are the more flexible unified source format**.

For the current candidates:
- D-Fire: cannot directly train either segmentation model.
- DataCluster Labs Kaggle listing: not approved until annotations are verified.
- Kaggle Fire/Smoke YOLO detection item: detection-only based on the available description.
- Hugging Face fire segmentation item: supports fire-only experiments, but generated labels and missing smoke class make it incomplete.
- DBSeg research lead: potentially suitable if its release, labels, splits, and license are verified.

## 7. License and attribution checklist

Before any data enters the project:
- Save the dataset card and license text/version.
- Record original image sources and any upstream licenses.
- Check whether academic research, model training, redistribution, and commercial deployment are separately permitted.
- Do not assume Kaggle availability means public-domain or unrestricted reuse.
- Record required citations/attribution in `docs/dataset_selection.md`.
- Keep source-specific restrictions attached to each image if datasets are mixed.
- Do not redistribute a repackaged dataset unless every source license permits it.

## 8. Research sources

1. D-Fire Kaggle: https://www.kaggle.com/datasets/shubhamkarande13/d-fire
2. D-Fire original repository and dataset description: https://github.com/gaia-solutions-on-demand/DFireDataset
3. D-Fire license: https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/LICENSE
4. D-Fire official project page: https://gaia-solutions-on-demand.github.io/DFireDataset/
5. DataCluster Labs Kaggle dataset: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
6. DataCluster Labs Kaggle profile: https://www.kaggle.com/dataclusterlabs/datasets
7. Kaggle Fire/Smoke YOLO dataset reference: https://www.kaggle.com/datasets/azimjaan21/fire-and-smoke-dataset-object-detection-yolo
8. Project describing bounding-box-to-rectangle pseudo-mask generation: https://github.com/phattien12/Smoke-Fire-Segmentation
9. Hugging Face fire segmentation dataset: https://huggingface.co/datasets/sreeharivp23/fire-segmentation-dataset
10. Hugging Face fire segmentation duplicate/preview: https://huggingface.co/datasets/Pravesh-21/fire-segmentation-dataset
11. Related model card and label provenance: https://huggingface.co/sreeharivp23/fire-segmentation-yolo11n
12. DBSeg research lead: https://ayazkhan789.github.io/DBSeg/
13. Kaggle dataset documentation: https://www.kaggle.com/docs/datasets
14. Published D-Fire dataset description: https://pmc.ncbi.nlm.nih.gov/articles/PMC11991653/

## 9. Research limitations and verification status

This is a public-source desk review, not a forensic inspection of downloaded Kaggle archives. Kaggle pages may hide file listings or metadata behind sign-in/dynamic rendering. Any value explicitly marked “not verified” must remain unknown until the dataset card and archive contents are inspected. No dataset was downloaded in this phase.

**Phase 01 research outcome:** D-Fire is verified as detection-only. No reviewed Kaggle dataset is yet verified to provide trustworthy, manually annotated, dual-class fire-and-smoke segmentation masks/polygons. Dataset approval should therefore be conditional on additional file-level verification or a decision to commission/perform manual segmentation annotation.
