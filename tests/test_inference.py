from io import BytesIO

from PIL import Image

from firesight.inference.predictor import Predictor
from firesight.models.segmentation import ModelNotAvailableError


def test_predictor_rejects_missing_checkpoint():
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (255, 40, 20)).save(buffer, format="PNG")
    try:
        Predictor("unet", "models/checkpoints/not-present.pt", image_size=8)
    except ModelNotAvailableError as exc:
        assert "checkpoint is unavailable" in str(exc)
    else:
        raise AssertionError("Missing checkpoint did not raise ModelNotAvailableError")

