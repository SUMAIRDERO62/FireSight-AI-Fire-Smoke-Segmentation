# FireSight AI

FireSight AI is a comparative deep learning project for fire and smoke image
segmentation. It targets the following segmentation models:

- U-Net
- YOLO11-seg

The primary dataset source is Kaggle. The application accepts image files only;
it does not support video files, webcams, streaming, or real-time monitoring.

## Scope

- Developer: Sumair Ahmed Dero
- Roll Number: 2K24/AIE/62
- Degree: BS Artificial Intelligence
- University: University of Sindh, Jamshoro
- Subject: Computer Vision

No model is trained or shipped with a checkpoint in this phase. The current
repository provides architecture, API contracts, tests, and image utilities.
It intentionally returns a clear "model unavailable" response instead of
generating fake segmentation masks.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -e ".[segmentation]"
python -m pytest -q
uvicorn app.backend.main:app --reload
```

The API exposes `GET /health`, `GET /models`, and `POST /segment`. Both
`U-Net` and `YOLO11-seg` use a shared segmentation interface. Generated
datasets, checkpoints, reports, and output masks are ignored by Git.

## Project layout

- `src/firesight`: reusable data, preprocessing, model, inference, training, evaluation, and visualization code.
- `scripts`: command-line entry points for dataset preparation, training, and evaluation.
- `app`: API backend and browser frontend.
- `configs`: YAML configuration for model and training.
- `tests`: focused unit and API tests.

