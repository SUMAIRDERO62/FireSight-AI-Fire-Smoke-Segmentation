from io import BytesIO

from PIL import Image

from firesight.inference.predictor import Predictor


def test_predictor_returns_known_class():
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (255, 40, 20)).save(buffer, format="PNG")
    result = Predictor(image_size=8).predict_bytes(buffer.getvalue())
    assert result.label in {"fire", "smoke", "normal"}
    assert 0 <= result.confidence <= 1

