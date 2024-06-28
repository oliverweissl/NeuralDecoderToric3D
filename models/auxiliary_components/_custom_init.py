from torch import nn
import torch


def custom_init(module: nn.Module) -> None:
    """
    Initialize Layer weights with custom init functions.

    :param module: The module ot initialize.
    """
    match module:
        case nn.Conv3d() | nn.Conv2d:
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_out")
        case nn.BatchNorm3d() | nn.BatchNorm2d:
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
