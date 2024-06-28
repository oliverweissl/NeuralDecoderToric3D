import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray


def categorical_accuracy(y_pred: Tensor, y_true: Tensor) -> tuple[float, float]:
    """
    Compute the categorical accuracy.

    :param y_pred: The prediction.
    :param y_true: The true label.
    :return: The accuracy and std.
    """
    _, predicted = torch.max(y_pred, 1)

    correct: Tensor = (predicted == y_true).float()
    correct: NDArray[np.float_] = correct.cpu().numpy()
    accuracy: float = correct.sum() / len(correct)
    std_accuracy = np.sqrt(accuracy * (1. - accuracy) / len(correct))
    return accuracy, std_accuracy
