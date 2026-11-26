
"""Local storage for FireSight AI prediction results."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ...core.config import RESULTS_DIR


def save_prediction_result(
    original_image: Image.Image,
    masks: dict[str, np.ndarray],
    overlay: Image.Image,
    metadata: dict,
) -> dict:
    """Save prediction artifacts and metadata in a unique directory."""

    result_id = uuid4().hex
    result_dir = Path(RESULTS_DIR) / result_id
    result_dir.mkdir(parents=True, exist_ok=False)

    original_path = result_dir / "original.png"
    fire_path = result_dir / "fire_mask.png"
    smoke_path = result_dir / "smoke_mask.png"
    overlay_path = result_dir / "overlay.png"
    metadata_path = result_dir / "metadata.json"

    original_image.convert("RGB").save(original_path, format="PNG")

    for class_name, path in (
        ("fire", fire_path),
        ("smoke", smoke_path),
    ):
        mask = masks[class_name]

        mask_image = Image.fromarray(
            (mask.astype(np.uint8) * 255),
            mode="L",
        )

        mask_image.save(path, format="PNG")

    overlay.convert("RGB").save(overlay_path, format="PNG")

    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"masks", "overlay"}
    }

    safe_metadata["result_id"] = result_id
    safe_metadata["created_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    metadata_path.write_text(
        json.dumps(safe_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "result_id": result_id,
        "result_directory": str(result_dir),
        "original_path": str(original_path),
        "fire_mask_path": str(fire_path),
        "smoke_mask_path": str(smoke_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(metadata_path),
    }


def get_result_path(result_id: str) -> Path:
    """Resolve a result directory while preventing path traversal."""

    if not re.fullmatch(r"[0-9a-f]{32}", result_id):
        raise ValueError("Invalid result ID.")

    result_dir = Path(RESULTS_DIR) / result_id

    if not result_dir.is_dir():
        raise FileNotFoundError("Prediction result not found.")

    return result_dir
