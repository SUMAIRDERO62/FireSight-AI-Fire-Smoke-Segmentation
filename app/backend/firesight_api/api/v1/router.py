"""Central API v1 router."""

from fastapi import APIRouter

from .controllers.health_controller import router as health_router
from .controllers.model_controller import router as model_router
from .controllers.prediction_controller import router as prediction_router
from .controllers.result_controller import router as result_router

router = APIRouter()

router.include_router(health_router)
router.include_router(model_router)
router.include_router(prediction_router)
router.include_router(result_router)
