
"""Validate uploaded images before FireSight AI inference."""

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..core.config import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
)


FORMAT_TO_CONTENT_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def validate_image(
    image_bytes: bytes,
    filename: str,
    content_type: str | None,
) -> Image.Image:
    """Validate uploaded image data and return an RGB PIL image."""

    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError("Image exceeds the maximum upload size of 10 MB.")

    extension = Path(filename or "").suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension: {extension or 'missing'}."
        )

    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {content_type or 'missing'}."
        )

    try:
        # Verify image integrity without decoding the full image.
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Reopen after verify() and decode the image.
        with Image.open(BytesIO(image_bytes)) as image:
            actual_format = image.format

            expected_content_type = FORMAT_TO_CONTENT_TYPE.get(
                actual_format
            )

            if expected_content_type != content_type:
                raise ValueError(
                    "Image content does not match its declared content type."
                )

            if actual_format == "JPEG" and extension not in {
                ".jpg", ".jpeg"
            }:
                raise ValueError("JPEG images must use .jpg or .jpeg.")

            if actual_format == "PNG" and extension != ".png":
                raise ValueError("PNG images must use the .png extension.")

            if actual_format == "BMP" and extension != ".bmp":
                raise ValueError("BMP images must use the .bmp extension.")

            if actual_format == "TIFF" and extension not in {
                ".tif", ".tiff"
            }:
                raise ValueError("TIFF images must use .tif or .tiff.")

            image.load()
            return image.convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Uploaded file is not a valid readable image.") from exc
