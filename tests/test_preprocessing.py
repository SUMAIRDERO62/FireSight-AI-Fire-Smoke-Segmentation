from PIL import Image
from io import BytesIO

from firesight.preprocessing.image import load_image


def test_load_image_returns_normalized_rgb_array():
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (255, 0, 0)).save(buffer, format="PNG")
    image = load_image(buffer.getvalue(), image_size=8)
    assert image.shape == (8, 8, 3)
    assert image.dtype.name == "uint8"

