# FireSight AI — Kaggle Page Verification Report
**Phase:** 01 — Dataset research and selection  
**Evidence date:** 21 November 2026  
**Evidence source:** User-provided copied text from Kaggle Data Explorer pages  
**Status:** Page metadata reviewed; archive annotation contents not yet inspected. No dataset downloaded.

## 1. D-Fire — Kaggle page evidence

Source: https://www.kaggle.com/datasets/shubhamkarande13/d-fire

The supplied Kaggle page shows:
- Publisher: Shubham Karande.
- Version 1; displayed size 3.12 GB.
- Kaggle license field: Apache 2.0.
- Data Explorer summary: 43.1K files.
- `test` contains `images` and `labels`; the displayed test directory has 4,306 image files and 4,306 label files.
- A `train` directory is also shown.
- No segmentation-mask directory or polygon evidence appears in the supplied listing.

### Interpretation

Matching image and label counts indicate paired files for the displayed test subset, but do not identify the annotation task. The original D-Fire documentation describes YOLO detection bounding boxes. Until sample label rows are inspected, classify this Kaggle copy as detection-only for FireSight AI.

The Kaggle page's Apache 2.0 field is distinct evidence from the original repository's licensing statements. The original repository warns that some source images remain subject to third-party terms. Preserve both records; do not assume the Kaggle field resolves upstream image rights.

### Current decision

**Supplementary image/detection dataset only. Not approved as segmentation data.**

### Remaining checks

1. Open one `test/labels/*.txt` file in Kaggle Data Explorer and inspect a few rows.
2. Five values per row (`class x_center y_center width height`) indicate YOLO detection.
3. A class ID followed by multiple coordinate pairs may indicate YOLO segmentation; verify the dataset's class documentation.
4. Look for actual mask files and their image-pairing convention.
5. Record train and test counts separately. Exact train counts and a validation split are not established by the supplied page text.

## 2. DataCluster Labs — Kaggle page evidence

Source: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset

The supplied page shows:
- Publisher: DataCluster Labs.
- Publisher claim: 7,000+ original fire and smoke images.
- Claimed 1,000+ contributors and 400+ cities across India.
- Publisher says images were manually reviewed by computer-vision professionals.
- Publisher advertises COCO, YOLO, Pascal VOC and TFRecord annotation formats.
- Kaggle version 3; displayed size 93.27 MB.
- An `Annotations` directory with 100 files.
- Page summary: 200 files.
- Kaggle license field: “Data files © Original Authors”.
- Publisher says full-dataset research/commercial access can be licensed by contacting `sales@datacluster.ai`.

### Interpretation

The 7,000+ figure describes the publisher's larger collection, not necessarily the Kaggle sample. The visible Kaggle sample summary is 200 files, including 100 annotation files. Do not report 7,000+ as the Kaggle archive's image count.

The advertised formats do not prove segmentation. COCO, YOLO and Pascal VOC each have detection-only variants. “Manually reviewed images” does not establish that object boundaries were manually polygon-annotated.

“Data files © Original Authors” is not a permissive license such as CC BY or Apache 2.0. Do not assume unrestricted redistribution or commercial use.

### Current decision

**Highest-priority candidate for annotation inspection, but not yet approved as segmentation data.**

### Remaining checks

1. Expand `Annotations` and record visible filenames/extensions.
2. Determine whether it contains `.json`, `.xml`, `.txt`, `.tfrecord`, mask images, or another format.
3. For COCO JSON: `bbox` alone means detection; non-empty `segmentation` polygon arrays or RLE indicate segmentation candidates.
4. For Pascal VOC XML: inspect whether it contains only `<bndbox>` or actual segmentation mask references/data.
5. For YOLO text: inspect the number and meaning of values per row.
6. Establish actual image count, class counts, image-to-annotation pairing and split structure.
7. Ask the publisher which rights apply to the Kaggle sample: academic research, model training, derivative annotations, public demonstration, commercial use and redistribution.

Publisher contact shown on the page: `sales@datacluster.ai`.

## 3. Current candidate comparison

| Criterion | D-Fire Kaggle | DataCluster Labs Kaggle |
|---|---|---|
| Page metadata reviewed | Yes | Yes |
| Displayed archive size | 3.12 GB | 93.27 MB |
| Displayed file count | 43.1K | 200 |
| Image/label pairing visible | Test: 4,306 each | Not established |
| Fire + smoke claimed | Yes, original project | Yes |
| Segmentation proven by page evidence | No | No |
| Detection annotations likely | Yes, original documentation | Possible; inspect files |
| Kaggle license field | Apache 2.0 | Data files © Original Authors |
| Current role | Supplementary detection data | Priority annotation inspection |

## 4. Exact next inspection sequence

Do not download the full D-Fire archive. Do not start training.

### First: inspect DataCluster's 100 annotation files

Open Kaggle Data Explorer and expand `Annotations`. Provide:
- visible filenames/extensions;
- a screenshot of the expanded directory;
- if Kaggle previews a JSON/XML/TXT file, copy a small excerpt of one annotation record.

We need to establish whether it contains actual polygons or masks.

### Second: inspect a D-Fire label

Open one file under `test/labels/` and copy 2–3 sample rows. Do not download the archive. This will verify the annotation row structure.

### Third: resolve licensing

For D-Fire, compare the Kaggle Apache 2.0 metadata with the original repository's source-image caveat.
For DataCluster, ask the publisher what rights apply to the Kaggle sample and what rights require a paid license.

## 5. Decision gate

Neither dataset is approved for U-Net or YOLO11-seg training yet.

- D-Fire remains supplementary detection data until actual labels prove otherwise and licensing is resolved.
- DataCluster is the leading candidate for further inspection, but no segmentation claim is accepted until a real polygon/RLE/mask annotation is inspected.

**No bulk download, preprocessing, generated masks, dataset merge, or training is authorized in this stage.**
