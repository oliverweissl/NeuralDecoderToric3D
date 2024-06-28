import torch
from torch import Tensor


def exact_match_ratio(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Compute the exact match ratio.

    :param y_pred: The prediction.
    :param y_true: The true label.
    :return: The accuracy.
    """
    match_tensor = torch.all(torch.eq(y_pred.round(), y_true), dim=1)
    emr = torch.mean(match_tensor.float())
    return emr.item()
