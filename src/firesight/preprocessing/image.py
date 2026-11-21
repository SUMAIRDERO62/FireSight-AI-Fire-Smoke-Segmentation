from io import BytesIO

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported decodable image."""


def load_image(data: bytes, image_size: int | None = None) -> np.ndarray:
    """Decode one image file and return an RGB array.

    Args:
        data: Encoded image bytes.
        image_size: Optional square target size.

    Raises:
        ValueError: If the data is not a decodable image.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB")
            if image_size is not None:
                image = image.resize((image_size, image_size))
            return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Uploaded data is not a valid image.") from exc

