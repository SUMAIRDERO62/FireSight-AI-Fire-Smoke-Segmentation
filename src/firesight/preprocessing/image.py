from io import BytesIO

import numpy as np
from PIL import Image


def load_image(data: bytes, image_size: int = 224) -> np.ndarray:
    """Decode bytes and return a normalized RGB tensor-like array."""
    with Image.open(BytesIO(data)) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0

