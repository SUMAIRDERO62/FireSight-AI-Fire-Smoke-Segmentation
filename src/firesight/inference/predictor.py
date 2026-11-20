from pathlib import Path

from firesight.models.baseline import BaselineClassifier, Prediction
from firesight.preprocessing.image import load_image


class Predictor:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.model = BaselineClassifier()

    def predict_bytes(self, data: bytes) -> Prediction:
        return self.model.predict(load_image(data, self.image_size))

    def predict_file(self, path: str | Path) -> Prediction:
        return self.predict_bytes(Path(path).read_bytes())

