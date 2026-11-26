# from fastapi import FastAPI, File, HTTPException, UploadFile

# from firesight.models.segmentation import (
#     ModelNotAvailableError,
# )
# from firesight.preprocessing.image import InvalidImageError

# app = FastAPI(title="FireSight AI Segmentation API", version="0.1.0")

# MODEL_CHECKPOINTS = {
#     "unet": "models/checkpoints/unet.pt",
#     "yolo11-seg": "models/checkpoints/yolo11-seg.pt",
# }


# @app.get("/health")
# def health() -> dict[str, str]:
#     return {"status": "ok"}


# @app.get("/models")
# def models() -> dict[str, list[str]]:
#     return {"models": list(MODEL_CHECKPOINTS)}


# @app.post("/segment")
# async def segment(
#     file: UploadFile = File(...),
#     model: str = "unet",
# ) -> dict:
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=415, detail="An image file is required")
#     if model not in MODEL_CHECKPOINTS:
#         raise HTTPException(status_code=400, detail="Unsupported segmentation model")
#     try:
#         from pathlib import Path

#         from firesight.inference.predictor import Predictor

#         predictor = Predictor(model, Path(MODEL_CHECKPOINTS[model]))
#         predictor.predict_bytes(await file.read())
#     except ModelNotAvailableError as exc:
#         raise HTTPException(status_code=503, detail=str(exc)) from exc
#     except InvalidImageError as exc:
#         raise HTTPException(status_code=415, detail=str(exc)) from exc

"""FireSight AI FastAPI application entry point."""

from fastapi import FastAPI

from app.backend.firesight_api.api.v1.router import router as api_router
from app.backend.firesight_api.core.config import (
    API_PREFIX,
    APP_NAME,
    APP_VERSION,
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Image-based fire and smoke segmentation using "
        "trained U-Net and YOLO11-seg models."
    ),
)

app.include_router(api_router, prefix=API_PREFIX)


@app.get("/", tags=["Root"])
def root():
    """Return basic API information."""

    return {
        "success": True,
        "application": APP_NAME,
        "version": APP_VERSION,
        "documentation": "/docs",
        "health": f"{API_PREFIX}/health",
    }
