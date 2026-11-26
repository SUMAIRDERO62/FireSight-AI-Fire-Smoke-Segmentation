
"""Central registry for FireSight AI segmentation models."""

from threading import Lock
from typing import Any

from ..core.config import UNET_CHECKPOINT, YOLO_CHECKPOINT
from ..infrastructure.model_loaders.unet_loader import load_unet_model
from ..infrastructure.model_loaders.yolo_loader import load_yolo_model


MODEL_LOADERS = {
    "unet": load_unet_model,
    "yolo11-seg": load_yolo_model,
}

MODEL_CHECKPOINTS = {
    "unet": UNET_CHECKPOINT,
    "yolo11-seg": YOLO_CHECKPOINT,
}

_models: dict[str, Any] = {}
_model_errors: dict[str, str] = {}
_registry_lock = Lock()


def get_model(model_name: str) -> Any:
    """Return a cached model, loading it on its first request."""

    if model_name not in MODEL_LOADERS:
        supported = ", ".join(MODEL_LOADERS)
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        )

    with _registry_lock:
        if model_name in _models:
            return _models[model_name]

        try:
            model = MODEL_LOADERS[model_name]()
            _models[model_name] = model
            _model_errors.pop(model_name, None)
            return model

        except Exception as exc:
            _model_errors[model_name] = str(exc)
            raise RuntimeError(
                f"Failed to load model '{model_name}': {exc}"
            ) from exc


def get_model_status() -> dict:
    """Return model availability and loading status without loading models."""

    status = {}

    for model_name, checkpoint in MODEL_CHECKPOINTS.items():
        status[model_name] = {
            "loaded": model_name in _models,
            "checkpoint_available": checkpoint.is_file(),
            "error": _model_errors.get(model_name),
        }

    return status


def get_supported_models() -> list[str]:
    """Return the supported model identifiers."""

    return list(MODEL_LOADERS.keys())
