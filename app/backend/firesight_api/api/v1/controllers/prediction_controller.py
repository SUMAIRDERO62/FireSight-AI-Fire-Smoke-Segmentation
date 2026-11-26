
"""Image prediction endpoint."""

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from ....services.prediction_service import predict_image
from ....services.result_service import store_prediction
from ....utils.image_validation import validate_image

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("unet"),
):
    """Upload an image and run fire/smoke segmentation."""

    try:
        image_bytes = await file.read()

        image = validate_image(
            image_bytes=image_bytes,
            filename=file.filename or "",
            content_type=file.content_type,
        )

        prediction = predict_image(
            image=image,
            model_name=model_name,
        )

        saved = store_prediction(
            original_image=image,
            prediction=prediction,
        )

        return {
            "success": True,
            "message": "Segmentation completed successfully.",
            "result_id": saved["result_id"],
            "model": prediction["model"],
            "image": {
                "width": prediction["image_width"],
                "height": prediction["image_height"],
            },
            "processing_time_seconds": prediction[
                "processing_time_seconds"
            ],
            "classes": prediction["classes"],
            "results": {
                "overlay": f"/api/v1/results/{saved['result_id']}/overlay",
                "fire_mask": f"/api/v1/results/{saved['result_id']}/fire-mask",
                "smoke_mask": f"/api/v1/results/{saved['result_id']}/smoke-mask",
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    finally:
        await file.close()
