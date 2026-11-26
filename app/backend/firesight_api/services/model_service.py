
"""Business logic for model information and availability."""

from ..models.model_registry import (
    get_model_status,
    get_supported_models,
)


MODEL_DETAILS = {
    "unet": {
        "name": "U-Net",
        "task": "semantic_segmentation",
        "classes": ["fire", "smoke"],
        "input_size": "256x256",
    },
    "yolo11-seg": {
        "name": "YOLO11-seg",
        "task": "instance_segmentation",
        "classes": ["fire", "smoke"],
        "input_size": "640",
    },
}


def list_models() -> list[dict]:
    """Return supported models and their availability."""

    statuses = get_model_status()

    return [
        {
            "id": model_id,
            **MODEL_DETAILS[model_id],
            **statuses[model_id],
        }
        for model_id in get_supported_models()
    ]


def get_model_information(model_name: str) -> dict:
    """Return information for one supported model."""

    if model_name not in MODEL_DETAILS:
        raise ValueError(f"Unsupported model: {model_name}")

    return next(
        model
        for model in list_models()
        if model["id"] == model_name
    )
