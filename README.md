# FireSight AI
### Intelligent Fire & Smoke Detection Using Computer Vision

<p align="center">
  <strong>Academic Computer Vision & Research Project</strong><br>
  Dataset Engineering | Deep Learning | Object Detection | Image Segmentation
</p>

---

## Project Overview

**FireSight AI** is an academic computer-vision project focused on developing and evaluating intelligent systems for fire and smoke detection and segmentation.

The project follows a structured research and engineering workflow, covering dataset investigation, annotation validation, model development, training pipelines, evaluation, and comparative analysis.

The primary objective is to investigate how deep-learning architectures can identify fire and smoke in images and, where suitable pixel-level annotations are available, perform segmentation.

Rather than focusing only on model implementation, FireSight AI emphasizes:

- Dataset quality and annotation reliability.
- Reproducible data preparation and experiments.
- Deep-learning model development.
- Automated testing and validation.
- Transparent evaluation and reporting.
- Responsible interpretation of experimental results.

> **Current research status:** Phase 03 U-Net development and Phase 04 YOLO11-seg pipeline implementation have been completed and tested locally. Full GPU training and final comparative evaluation remain pending.

---

## Author & Project Information

| Field | Details |
|---|---|
| Project Name | FireSight AI |
| Author | **Sumair Ahmed Dero** |
| Role | Project Developer & Researcher |
| Academic Background | BS Artificial Intelligence |
| Institution | University of Sindh, Jamshoro |
| Project Category | Academic / Research-oriented Computer Vision |
| Primary Domain | Artificial Intelligence & Deep Learning |
| Programming Language | Python |
| Repository | FireSight AI — Fire & Smoke Segmentation |

### Academic Project Description

This project is developed as an academic and research-oriented computer-vision initiative. It explores the application of deep-learning techniques to fire and smoke recognition, with particular attention to dataset engineering, model architecture, segmentation methods, and experimental evaluation.

---

## Project Objectives

1. **Fire and Smoke Recognition** — Investigate automated recognition of fire and smoke in images.
2. **Dataset Engineering** — Inspect, validate, organize, and prepare image datasets and annotations.
3. **Deep Learning Models** — Develop and investigate U-Net and YOLO-based architectures.
4. **Image Segmentation** — Explore pixel-level fire and smoke segmentation using genuine segmentation annotations.
5. **Model Evaluation** — Compare models using appropriate quantitative and qualitative metrics.
6. **Reproducibility** — Maintain configurations, deterministic data splits, automated tests, and experiment reports.
7. **Research Documentation** — Record methodology, limitations, findings, and future improvements.

---

## Development Progress

FireSight AI is being developed in independent phases.

| Phase | Description | Status |
|---|---|---|
| Phase 01 | Project foundation and initial data engineering | Completed |
| Phase 02 | Dataset inspection, annotation audit, and data preparation | Locally completed; quality follow-ups documented |
| Phase 03 | U-Net model development and training/evaluation pipeline | Implementation completed; tests passed |
| Phase 04 | YOLO11 segmentation pipeline, evaluation, and prediction | Implementation and local testing completed |
| Phase 05 | U-Net vs YOLO11 comparative evaluation | Pending |
| Phase 06 | Final research analysis and documentation | Planned |

### Phase 02 — Dataset Engineering

Work included:

- D-Fire image and label pairing inspection.
- DataCluster annotation-format inspection.
- Dataset manifests and annotation statistics.
- Reproducible train, validation, and test preparation.
- Duplicate and annotation-quality investigations.

Important data-quality findings:

- D-Fire annotations inspected during the initial audit used YOLO bounding-box format.
- DataCluster annotations inspected during the audit used Pascal VOC XML bounding boxes.
- Some invalid D-Fire annotation rows were identified during follow-up diagnostics.
- The authoritative mapping of D-Fire numeric class IDs requires verification.

**Important:** Bounding-box annotations are not segmentation masks. They must not be treated as pixel-level ground truth.

### Phase 03 — U-Net Development

The U-Net development pipeline includes:

- U-Net model implementation.
- Training workflow.
- Evaluation components.
- Prediction and visualization support.
- Automated tests.

The Phase 03 implementation was completed and recorded in the project Git history.

### Phase 04 — YOLO11 Segmentation Pipeline

The YOLO11-seg development work includes:

- YOLO annotation auditing and visualization.
- Dataset preparation and split-safety utilities.
- Duplicate-pair verification and dataset analysis.
- YOLO11 segmentation configuration.
- Model and training components.
- Evaluation and prediction scripts.
- Automated Phase 04 tests.

**Local verification:** The complete test suite reported 70 passing tests. One dependency deprecation warning was reported; it did not cause a test failure.

Prediction and evaluation workflows were also smoke-tested locally.

**Training qualification:** The currently tested YOLO checkpoint was produced through a one-epoch CPU smoke test. It verifies pipeline execution, not final model performance. Full GPU training is still required.

---

## Models and Technologies

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Object Detection / Segmentation | Ultralytics YOLO11 |
| Segmentation Architecture | U-Net |
| API Framework | FastAPI |
| Image Processing | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Configuration | YAML |
| Testing | Pytest |
| Version Control | Git and GitHub |

---

## Project Structure

```text
FireSight-AI/
│
├── configs/
│   └── yolo11_seg.yaml
│
├── data/
│   ├── raw/                 # Original datasets (not committed)
│   ├── interim/             # Temporary data (not committed)
│   ├── processed/           # Prepared datasets (not committed)
│   └── annotations/         # Annotation data (not committed)
│
├── docs/                    # Phase documentation
│
├── models/                  # Model-related resources
│
├── reports/                 # Evaluation and research reports
│
├── scripts/
│   ├── audit_yolo_annotations.py
│   ├── create_annotation_contact_sheets.py
│   ├── create_split_safe_dataset.py
│   ├── evaluate_yolo11_seg.py
│   ├── prepare_yolo_dataset.py
│   ├── predict_yolo11_seg.py
│   ├── train_yolo11_seg.py
│   └── visualize_yolo_annotations.py
│
├── src/
│   └── firesight/
│       ├── data/
│       ├── evaluation/
│       ├── models/
│       └── training/
│
├── tests/
│   ├── phase03/
│   └── phase04/
│
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── .gitignore
└── README.md
```

Generated datasets, model weights, and experiment outputs are intentionally excluded from version control.

---

## Installation & Setup

### 1. Clone the repository

```powershell
git clone https://github.com/SUMAIRDERO62/FireSight-AI-Fire-Smoke-Segmentation.git

cd FireSight-AI-Fire-Smoke-Segmentation
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Verify Python and dependencies

```powershell
python --version
python -m pip list
```

For GPU training, install a PyTorch build compatible with the target machine's CUDA environment and verify GPU availability before starting experiments.

---

## Running Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

For Phase 04 tests:

```powershell
python -m pytest tests/phase04 -q
```

For Phase 03 tests:

```powershell
python -m pytest tests/phase03 -q
```

The reported local full-suite result for the current implementation was:

```text
70 passed
```

Test results verify the tested software behavior; they do not establish final model accuracy or dataset correctness.

---

## Experimental Methodology

The planned comparative evaluation will use a consistent experimental methodology.

### Quantitative Metrics

- Dice coefficient.
- Intersection over Union (IoU).
- Per-class IoU.
- Precision and recall.
- Inference time per image.
- Model size and computational requirements.
- YOLO mask mAP where applicable.

Detection mAP and pixel-level segmentation IoU measure different tasks and will not be treated as interchangeable metrics.

### Qualitative Evaluation

The research will also inspect:

- Correctly segmented fire and smoke regions.
- Missed detections and false positives.
- Boundary quality.
- Small or partially visible objects.
- Difficult lighting and background conditions.
- Differences between U-Net and YOLO11 predictions.

No final model winner will be declared without completed, comparable experiments.

---

## Dataset & Research Integrity

The following principles guide this project:

1. Preserve original datasets.
2. Do not commit raw or processed datasets without appropriate permission.
3. Do not invent annotation class mappings.
4. Do not convert bounding boxes into segmentation masks and present them as ground truth.
5. Record malformed annotations and data exclusions.
6. Keep train, validation, and test splits reproducible.
7. Investigate duplicate and near-duplicate data leakage.
8. Avoid tuning on the independent test set.
9. Record hardware, configuration, preprocessing, and evaluation methodology.
10. Report limitations and unsuccessful experiments honestly.

Dataset licensing and redistribution permissions must be reviewed before publishing dataset contents or sample images.

---

## Roadmap

### Completed / Locally Verified

- [x] Initial project foundation.
- [x] Dataset inspection and engineering workflows.
- [x] U-Net model development pipeline.
- [x] YOLO11 segmentation pipeline implementation.
- [x] Evaluation and prediction scripts.
- [x] Automated local test suite verification.

### In Progress / Planned

- [ ] Final review of dataset class semantics.
- [ ] Final annotation-quality verification.
- [ ] Full U-Net training on suitable GPU hardware.
- [ ] Full YOLO11-seg training on suitable GPU hardware.
- [ ] Evaluation using the same independent test set.
- [ ] U-Net vs YOLO11 quantitative comparison.
- [ ] Qualitative prediction comparison.
- [ ] Failure-case analysis.
- [ ] Final research report and conclusions.

---

## Limitations

The current implementation should be interpreted in the context of its development stage.

- Full GPU training has not yet been completed.
- The CPU smoke-test checkpoint is not a final research model.
- Initial bounding-box datasets do not establish segmentation readiness.
- Final performance metrics have not yet been established.
- Comparative conclusions require completed experiments under consistent conditions.

These limitations are documented to distinguish verified engineering progress from future research objectives.

---

## Version Control

This project uses Git and GitHub to maintain a structured development history.

Example workflow:

```powershell
git status --short
git diff --check
git add README.md
git commit -m "docs: improve project documentation"
git push origin main
```

Only reviewed source code, tests, configuration, and documentation should be committed. Datasets, model weights, generated outputs, credentials, and local environment files must remain excluded.

---

## Project Vision

The long-term vision of FireSight AI is to develop a reliable and reproducible computer-vision workflow for fire and smoke analysis.

The project aims to combine sound dataset engineering, deep-learning model development, meaningful evaluation, and transparent research practices.

**Built with Python, PyTorch, Computer Vision, and a research-first mindset.**

---

<p align="center">
  <strong>FireSight AI</strong><br>
  Developed by <strong>Sumair Ahmed Dero</strong><br>
  Academic Computer Vision Project
</p>
