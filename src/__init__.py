"""Various Classes for experiments."""
from ._evaluation import evaluate_decoder
from ._trainer import Trainer
from ._trainer import Trainer
from ._data_generator import DataGenerator
from ._auxiliary_functions import generate_syndrome, sample_errors

__all__ = ["evaluate_decoder", "Trainer", "DataGenerator", "Trainer"]
