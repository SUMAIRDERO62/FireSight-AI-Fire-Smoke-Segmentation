"""Endpoints for retrieving saved prediction results."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ....services.result_service import retrieve_prediction

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("/{result_id}")
def get_result(result_id: str):
    """Return saved prediction metadata."""

    try:
        return {
            "success": True,
            **retrieve_prediction(result_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _get_image_response(result_id: str, filename: str):
    """Safely return a saved image artifact."""

    try:
        result = retrieve_prediction(result_id)
        image_path = Path(result["result_directory"]) / filename

        if not image_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Requested result image was not found.",
            )

        return FileResponse(
            path=image_path,
            media_type="image/png",
            filename=filename,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{result_id}/overlay")
def get_overlay(result_id: str):
    return _get_image_response(result_id, "overlay.png")


@router.get("/{result_id}/fire-mask")
def get_fire_mask(result_id: str):
    return _get_image_response(result_id, "fire_mask.png")


@router.get("/{result_id}/smoke-mask")
def get_smoke_mask(result_id: str):
    return _get_image_response(result_id, "smoke_mask.png")


@router.get("/{result_id}/original")
def get_original_image(result_id: str):
    return _get_image_response(result_id, "original.png")
