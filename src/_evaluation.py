from panqec.codes import StabilizerCode
from .metrics import categorical_accuracy
import torch
from ._data_generator import DataGenerator
from torch import nn
import numpy as np
from time import time


def evaluate_decoder(
        model: nn.Module,
        code: StabilizerCode,
        error_rate: float,
        trials: int = 100_000,
        batch_size: int = 512,
) -> tuple[float, float, list[float]]:
    """
    Evaluate the Neural Decoders performance.

    :param model: The model.
    :param code: The stabilizer code.
    :param error_rate: The error rate.
    :param trials: The amount of trials.
    :param batch_size: The batch size.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_generator = DataGenerator(code=code, verbose=False, error_rate=error_rate, batch_size=batch_size)

    runtimes = []
    accuracies = []
    for _ in range(trials // batch_size):
        X, y = data_generator.generate_batch(use_qmc=False, device=device)
        start = time()
        with torch.no_grad():
            y_pred = model(X)
        runtimes.append((time() - start)/batch_size)
        acc, _ = categorical_accuracy(y_pred, y)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    return accuracies.mean(), accuracies.std(), runtimes