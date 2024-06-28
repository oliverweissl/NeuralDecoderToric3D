"""Custom Loss Functions."""
from ._dynamic_ce_loss import DynamicCELoss
from ._dynamic_bce_loss import DynamicBCELoss

__all__ = ["DynamicBCELoss", "DynamicCELoss"]