from typing import Sequence


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    return sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)

