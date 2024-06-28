from typing import Callable
from ._categorical_accuracy import categorical_accuracy
from ._exact_match_ratio import exact_match_ratio


def get_eval_method(mode: str) -> Callable:
    """
    Get the corresponding metrics measure for task.

    :param mode: the type of measure wanted.
    :return: The metrics function.
    :raises ValueError: If the mode is not defined.
    """
    match mode:
        case "categorical":
            return categorical_accuracy
        case "exact":
            return exact_match_ratio
        case _:
            raise ValueError("No such metrics defined. ", mode)
