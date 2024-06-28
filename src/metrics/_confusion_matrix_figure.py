from torch import Tensor
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def confusion_matrix_figure(y_pred: Tensor, y_true: Tensor) -> Figure:
    """
    Get confusion matrix from predictions.

    :param y_pred: The prediction. (b, num_classes).
    :param y_true: The ground truth.
    :returns: A confusion matrix.
    """
    _, classes = y_pred.shape
    _, predicted = torch.max(y_pred, 1)

    predicted = predicted.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    confusion = confusion_matrix(y_true, predicted, labels=range(classes))
    display = ConfusionMatrixDisplay(confusion, display_labels=range(classes))
    fig, ax = plt.subplots(1, 1, figsize=(32, 32))
    display.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.close(fig)
    return fig
