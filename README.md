FireSight AI
Intelligent Fire & Smoke Detection — Dataset Engineering and Computer Vision Project
FireSight AI is a computer-vision project being developed to investigate fire and smoke recognition from images and, in later stages, support segmentation-based analysis. The project is being built in structured phases, with emphasis on dataset verification, reproducibility, annotation quality, testing, and transparent reporting.
Current verified stage: Phase 02 dataset audit and D-Fire detection-data preparation have been run locally. The prepared labels are YOLO object-detection bounding boxes, not segmentation masks. Model training and segmentation readiness are not claimed as complete.

Table of contents

- Project objectives
- Current development status
- Datasets reviewed
- Annotation quality findings
- Project structure
- Environment setup
- Run the dataset audit
- Prepare D-Fire detection data
- Run tests
- Data and annotation policy
- Git workflow
- Roadmap
  Project objectives
  The long-term project direction includes:

1. Fire and smoke recognition: develop computer-vision models that can identify fire and smoke in visual data.
2. Object detection: use bounding-box annotations for detection experiments where class semantics and label quality are verified.
3. Semantic segmentation: investigate a U-Net-style architecture after genuine pixel-level masks are available.
4. Instance segmentation: investigate YOLO segmentation models after polygon or instance-mask annotations are available.
5. Reproducible engineering: maintain dataset manifests, deterministic splits, validation checks, automated tests, and clear experiment records.
   These are project objectives, not claims that all models or capabilities have already been implemented.
   Current development status
   Area Current status
   Dataset research and inspection Completed for the datasets described below
   D-Fire image/label pairing audit 21,527 matching image-label records reported
   D-Fire annotation format YOLO object-detection boxes
   D-Fire class-name mapping Not yet verified from an authoritative source
   D-Fire invalid annotation review 26 invalid rows identified by local diagnostic
   D-Fire train/validation/test preparation Prepared locally; counts below
   Exact train/test image-hash overlap 0 reported by preparation script
   DataCluster annotation format Pascal VOC XML bounding boxes
   Phase 02 automated tests 10 tests passed in the reported local run
   U-Net ground-truth masks Not verified / not available in reviewed annotations
   YOLO segmentation polygons Not verified / not available in reviewed annotations
   Model training and evaluation Not yet reported as completed

D-Fire prepared split
Split Images
Train 14,585
Validation 2,636
Test 4,306
Total 21,527

The original test partition was retained, while validation was carved from the original training partition using seed 42 and validation ratio 0.15.

Data-quality qualification: The preparation output reported zero excluded records, but a subsequent raw-label diagnostic identified 26 invalid annotation rows. The prepared copy must not be treated as fully label-clean until the preparation pipeline is updated to detect and handle those rows and the reports are regenerated.
Datasets reviewed

1. D-Fire
   The local audit reported:

- 21,527 JPG images.
- 21,527 TXT label files.
- 21,527 matching image-label records.
- YOLO detection rows with five fields: class_id x_center y_center width height.
- Raw class-row counts: class 0 = 11,865 and class 1 = 14,692.
- No authoritative class-name mapping file was found in the inspected dataset directory.
  The numeric class IDs remain unresolved. Do not assume class 0 means fire or class 1 means smoke until the publisher's authoritative mapping is verified.

2. DataCluster
   The local audit reported:

- 100 JPG images.
- 34 Pascal VOC XML annotation files.
- 34 matching image/XML records.
- 66 images without a same-stem XML annotation.
- 0 orphan XML files.
- 40 annotated objects with class name fire.
- The inspected XML format uses <bndbox> coordinates and is detection annotation, not pixel-level segmentation.
  The 66 unmatched images must not automatically be interpreted as negative/background examples.
  Annotation quality findings
  The raw D-Fire diagnostic reported:
  Class ID Raw rows Valid rows Difference
  0 11,865 11,846 19
  1 14,692 14,685 7
  Total 26,557 26,531 26

The follow-up classification found these conditions. Categories overlap because a row can violate more than one rule:

- 17 rows with zero or negative width.
- 17 rows with zero or negative height.
- 8 rows with at least one coordinate/dimension outside the [0, 1] range.
- 26 unique invalid rows based on the raw-versus-valid row difference.
  The original files under data/raw/ must remain unchanged. Invalid rows should be logged in a machine-readable quality report. Do not silently clip coordinates, invent boxes, or convert bounding boxes into segmentation masks.
  Project structure
  The repository may evolve. This is the intended organization of relevant components:
  FireSight-AI/
  ├── configs/ # Project and experiment configuration
  ├── data/
  │ ├── raw/ # Original datasets (not committed)
  │ ├── interim/ # Temporary data products (not committed)
  │ ├── processed/ # Prepared datasets (not committed)
  │ └── annotations/ # Annotation work (not committed)
  ├── docs/ # Architecture, methodology and phase guides
  ├── models/
  │ └── checkpoints/ # Model weights (not committed)
  ├── reports/
  │ ├── phase02/ # Dataset audit reports
  │ ├── figures/ # Generated figures (not committed)
  │ └── results/ # Experiment outputs (not committed)
  ├── scripts/ # Audit and preparation entry points
  ├── src/
  │ └── firesight/ # Reusable Python package
  ├── tests/ # Automated tests
  ├── .gitignore
  ├── pyproject.toml
  ├── pytest.ini
  ├── requirements.txt
  └── README.md
  This is a documentation map, not a guarantee that every directory is already present locally.
  Environment setup
  Requirements:
- Python 3.12 (the reported local environment used Python 3.12.10)
- Git
- A virtual environment
- Datasets placed in the expected local data/raw/ structure
  Windows PowerShell
  From the repository root:
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python --version
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  If the optional Phase 02 dependency file is present and needed:
  python -m pip install -r requirements-phase02.txt
  Review both dependency files first if they contain package pins that may conflict.
  Run the dataset audit
  With the virtual environment active:
  python scripts/phase02_audit.py --project-root .
  The Phase 02 audit writes reports under reports/phase02/. Reported outputs include:
- dataset_audit.json
- dataset_manifest.csv
- annotation_class_counts.csv
- duplicate_groups.csv
- dataset_report.md
  Review these outputs before using data in a training workflow.
  Prepare D-Fire detection data
  The locally used command was:
  python scripts/prepare_phase02.py --project-root . --dataset d-fire --val-ratio 0.15 --seed 42
  Reported output directory:
  data/processed/phase02/d-fire-detection/
  ├── images/
  │ ├── train/
  │ ├── val/
  │ └── test/
  ├── labels/
  │ ├── train/
  │ ├── val/
  │ └── test/
  ├── preparation_report.json
  └── split_manifest.csv
  This is a detection-data layout, not a U-Net mask dataset and not a YOLO segmentation dataset.

Before training, update or verify the preparation workflow so it reports and handles malformed/invalid annotation rows. Re-run preparation and tests after that change.
Run tests
python -m pytest tests/phase02 -q
The reported local result for the current Phase 02 test suite was:
10 passed
Passing tests verify the tested code paths; they do not prove all source annotations are correct or that segmentation is ready.
Data and annotation policy

1. Preserve original dataset files under data/raw/.
2. Do not commit raw or processed datasets to Git.
3. Do not claim segmentation readiness based on bounding boxes.
4. Do not invent class-name mappings.
5. Do not automatically classify unmatched images as negative examples.
6. Record malformed labels and exclusions in reports.
7. Keep data splitting reproducible.
8. Review exact and near-duplicate leakage risks.
9. Verify dataset licensing and redistribution permissions before publishing data or sample images.
10. Keep credentials, API keys, and private configuration out of the repository.
    Git workflow
    Check changes before staging:
    git status --short
    git diff --check
    git diff
    Stage only reviewed source, test, configuration, and documentation files. Do not use git add . until confirming datasets, generated outputs, secrets, and large archives are ignored.

Example focused staging:
git add .gitignore README.md
git add scripts/phase02_audit.py scripts/prepare_phase02.py
git add src/firesight/data tests/phase02
git add docs/PHASE_02_DATA_ENGINEERING.md requirements-phase02.txt
Inspect staged changes:
git diff --cached --check
git diff --cached --stat
git status --short
Commit only after reviewing the staged diff.
Roadmap
Completed or locally verified

- [x] Initial dataset inspection and pairing audit for D-Fire and DataCluster
- [x] Detection annotation formats identified from inspected labels
- [x] D-Fire deterministic train/validation/test preparation run
- [x] Phase 02 test suite passed in the reported local run
- [x] D-Fire invalid annotation rows identified by follow-up diagnostic
      Pending
- [ ] Verify D-Fire class IDs using authoritative publisher documentation
- [ ] Update preparation pipeline to detect and handle invalid labels
- [ ] Regenerate prepared data and quality reports
- [ ] Investigate the 66 DataCluster images without matching XML
- [ ] Review dataset licenses and publication permissions
- [ ] Create or obtain genuine fire/smoke segmentation masks or polygons
- [ ] Perform visual annotation quality review
- [ ] Train and evaluate detection/segmentation models
- [ ] Record experiment configurations and metrics
      Project status statement
      FireSight AI has an audited and locally prepared object-detection data workflow under development. The current reviewed annotations are bounding boxes. The project does not yet claim completed model training, verified class semantics, or U-Net/YOLO segmentation readiness. Those milestones require additional source verification, data-quality handling, and genuine segmentation annotations.
