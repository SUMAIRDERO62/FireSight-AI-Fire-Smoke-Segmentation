
"""Business logic for saving and retrieving prediction results."""

import json

from ..infrastructure.storage.local_storage import (
    get_result_path,
    save_prediction_result,
)


def store_prediction(
    original_image,
    prediction: dict,
) -> dict:
    """Persist prediction artifacts and return result references."""

    saved = save_prediction_result(
        original_image=original_image,
        masks=prediction["masks"],
        overlay=prediction["overlay"],
        metadata={
            key: value
            for key, value in prediction.items()
            if key not in {"masks", "overlay"}
        },
    )

    return {
        "result_id": saved["result_id"],
        "overlay_path": saved["overlay_path"],
        "fire_mask_path": saved["fire_mask_path"],
        "smoke_mask_path": saved["smoke_mask_path"],
        "original_path": saved["original_path"],
        "metadata_path": saved["metadata_path"],
    }


def retrieve_prediction(result_id: str) -> dict:
    """Retrieve saved prediction metadata."""

    result_dir = get_result_path(result_id)
    metadata_path = result_dir / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError("Prediction metadata not found.")

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return {
        "result_id": result_id,
        "metadata": metadata,
        "result_directory": str(result_dir),
    }
