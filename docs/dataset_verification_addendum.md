# FireSight AI — Phase 01 Verification Addendum
**Date:** 21 November 2026  
**Purpose:** Corrections and additional publisher-level findings after approval of Option A.

## 1. D-Fire — publisher-level confirmation

Original repository: https://github.com/gaia-solutions-on-demand/DFireDataset  
Kaggle: https://www.kaggle.com/datasets/shubhamkarande13/d-fire

The original repository confirms:
- More than 21,000 images.
- 1,164 only-fire images.
- 5,867 only-smoke images.
- 4,658 fire-and-smoke images.
- 9,838 images with neither class.
- 14,692 fire bounding boxes and 11,865 smoke bounding boxes.
- YOLO normalized-coordinate annotations.
- Pre-split train, validation and test downloads are offered.

**Segmentation decision:** Detection-only. The publisher describes YOLO bounding boxes, not polygon segmentation or pixel masks.

**License caveat:** The repository has a license file, but it also explicitly says that images from third-party sources (including HPWREN, FESB/Wildfire Observers and FireNet) remain subject to their original providers' terms. The repository-level license must not be assumed to override those source-specific terms. Verify the license text and upstream terms before reuse.

## 2. DataCluster Labs — publisher-level confirmation

Kaggle: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset  
Publisher repository: https://github.com/datacluster-labs/Fire-and-Smoke-Dataset

The publisher's public README states:
- 7,000+ original fire and smoke images.
- Images collected/crowdsourced from 400+ urban and rural areas in India.
- 98% HD resolution or above (1920×1080 and above).
- Images manually reviewed and verified by the publisher's computer-vision professionals.
- Annotation formats offered: COCO, YOLO, Pascal VOC and TFRecord.
- Full dataset access is available by contacting the publisher; a Kaggle sample is linked.

**Critical distinction:** The README's phrase “manually reviewed and verified” refers to image review; it does not establish that segmentation masks were manually drawn. Listing COCO/YOLO/Pascal VOC/TFRecord also does not prove segmentation: those formats can store bounding boxes only. Inspect actual annotation records.

**Current decision:** Candidate for further inspection, not yet a segmentation dataset. The exact Kaggle sample image count, file list, annotation schema, class counts, split structure and Kaggle license were not readable from the public dynamic Kaggle page during this review.

Publisher contact shown in its README: sales@datacluster.ai. A licensing/access inquiry may be required for the full collection.

## 3. Hugging Face fire segmentation candidate — clarified scope

Dataset: https://huggingface.co/datasets/sreeharivp23/fire-segmentation-dataset

The dataset card describes:
- 1,348 images: 1,146 train and 202 validation.
- 1,098 fire-positive images and 250 negatives.
- One segmentation class: fire.
- YOLO segmentation polygon labels.
- Machine-generated labels: a pretrained YOLOv8 detector, SAM 2.1, contour polygonisation, and an HSV fallback for some images.
- Mixed-source licensing; upstream image terms must be checked.

**Decision:** It is not a Kaggle dataset, does not establish smoke segmentation, and its labels are not human-annotated ground truth. Keep it as a small experimental reference only.

## 4. DBSeg — availability correction

Project page: https://ayazkhan789.github.io/DBSeg/

The page reports a 5,612-image fire-and-smoke instance-segmentation benchmark with 13,115 polygon-annotated instances. However, its page labels the paper, code and dataset as “Soon”.

**Decision:** Treat it as a research lead, not as an available downloadable dataset. Do not count its images or polygons as accessible project data until a public archive, annotation files, license and splits can be verified.

## 5. What we still need to verify on Kaggle

For each Kaggle candidate, capture the following from the Kaggle dataset page without downloading the full archive:

1. Dataset title, owner, version/update date.
2. Dataset license shown on the Kaggle page.
3. File Explorer listing: file names, extensions, folder names and sizes.
4. Any README, YAML, JSON, XML or CSV annotation documentation visible in the listing.
5. Whether files suggest `masks/`, `annotations/`, COCO JSON, polygon arrays, or only YOLO `.txt` box labels.
6. Published train/validation/test folder structure, if present.

A file name such as `masks/` is only a clue; inspect sample mask values and alignment before calling it a true pixel mask. For COCO JSON, inspect whether each annotation has a non-empty `segmentation` polygon/RLE field or only `bbox`.

## 6. Approval gate remains unchanged

Option A is approved. This addendum does not authorize large dataset downloads, bulk preprocessing, annotation generation, or model training.

The next decision is based on verifiable annotation records and license terms—not on dataset marketing descriptions.
