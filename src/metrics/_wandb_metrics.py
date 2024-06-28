from __future__ import annotations
from dataclasses import dataclass
from matplotlib.figure import Figure
from torch import Tensor
from ._categorical_accuracy import categorical_accuracy
from ._confusion_matrix_figure import confusion_matrix_figure


@dataclass
class WandbMetrics:
    """Metrics of the trained model."""
    loss: float
    accuracy: float
    accuracy_std: float
    confusion_matrix: Figure
    learning_rate: float
    epoch_duration: float

    @staticmethod
    def get_metrics(y_pred: Tensor, y_true: Tensor, loss: float, learning_rate: float, epoch_duration: float) -> WandbMetrics:
        """
        Get metrics object from prediction.

        :param y_pred: The prediction.
        :param y_true: The ground truth.
        :param loss: The loss.
        :param learning_rate: The learning rate used.
        :param epoch_duration: The duration of an epoch.
        :returns: The metrics object.
        """
        accuracy, accuracy_std = categorical_accuracy(y_pred, y_true)
        cm = confusion_matrix_figure(y_pred, y_true)

        instance = WandbMetrics(
            loss=loss,
            accuracy=accuracy,
            accuracy_std=accuracy_std,
            confusion_matrix=cm,
            learning_rate=learning_rate,
            epoch_duration=epoch_duration
        )
        return instance
