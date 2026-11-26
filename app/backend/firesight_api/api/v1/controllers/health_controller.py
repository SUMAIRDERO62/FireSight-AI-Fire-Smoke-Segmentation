
"""Health endpoints for FireSight AI."""

from fastapi import APIRouter

from ....core.config import APP_NAME, APP_VERSION
from ....models.model_registry import get_model_status

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """Return API health and model checkpoint availability."""

    model_status = get_model_status()

    checkpoints_ready = all(
        item["checkpoint_available"]
        for item in model_status.values()
    )

    return {
        "success": True,
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "healthy" if checkpoints_ready else "degraded",
        "models": model_status,
    }
