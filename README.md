# FireSight AI

Starter project for fire and smoke detection from images.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
uvicorn app.backend.main:app --reload
```

The API exposes `GET /health` and `POST /predict`. The initial predictor is a
deterministic baseline so the project runs before a trained checkpoint exists.

## Project layout

- `src/firesight`: reusable data, preprocessing, model, inference, training, evaluation, and visualization code.
- `scripts`: command-line entry points for dataset preparation, training, and evaluation.
- `app`: API backend and browser frontend.
- `configs`: YAML configuration for model and training.
- `tests`: focused unit and API tests.

