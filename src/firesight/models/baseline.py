from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class BaselineClassifier:
    """Small image-statistics baseline used until a trained model is available."""

    classes = ("fire", "smoke", "normal")

    def predict(self, image: np.ndarray) -> Prediction:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        red_mean = float(image[..., 0].mean())
        brightness = float(image.mean())
        if red_mean > 0.58 and red_mean > float(image[..., 1].mean()) * 1.15:
            probabilities = {"fire": 0.80, "smoke": 0.10, "normal": 0.10}
        elif brightness < 0.35:
            probabilities = {"fire": 0.10, "smoke": 0.75, "normal": 0.15}
        else:
            probabilities = {"fire": 0.10, "smoke": 0.15, "normal": 0.75}
        label = max(probabilities, key=probabilities.get)
        return Prediction(label, probabilities[label], probabilities)

