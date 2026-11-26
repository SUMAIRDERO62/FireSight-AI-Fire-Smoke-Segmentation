
"""Model information endpoints."""

from fastapi import APIRouter, HTTPException

from ....services.model_service import (
    get_model_information,
    list_models,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
def get_models():
    """List supported segmentation models."""

    return {
        "success": True,
        "models": list_models(),
    }


@router.get("/{model_name}")
def get_model_details(model_name: str):
    """Return information about one supported model."""

    try:
        model = get_model_information(model_name)
        return {
            "success": True,
            "model": model,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
