from torch import nn, Tensor
from typing import Callable
from ..auxiliary_components import custom_init


class WideResBlock(nn.Module):
    """A wide residual block (inspired by WRN architecture)."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 layer_type: Callable,
                 norm_type: Callable
                 ) -> None:
        """
        Initialize the Wide Residual Block.

        :param in_channels: The channels in the input.
        :param out_channels: The channels in the output.
        :param kernel_size: The kernel size.
        :param layer_type: The type of layer to use.
        :param norm_type: The type of normalization layer.
        """
        super(WideResBlock, self).__init__()
        """Define Layers."""
        self.block = nn.Sequential(
            layer_type(in_channels, out_channels, kernel_size=kernel_size),
            nn.GELU(),
            layer_type(out_channels, out_channels, kernel_size=kernel_size),
        )

        self.batch_norm = norm_type(in_channels)
        self.non_linear = nn.GELU()
        self.skip = nn.Identity() if in_channels == out_channels else layer_type(in_channels, out_channels,
                                                                                 kernel_size=1)

        """Initialize Layer weights."""
        self.block.apply(custom_init)
        self.apply(custom_init)

    def forward(self, x: Tensor) -> Tensor:
        """Initial Processing."""
        x = self.batch_norm(x)
        x = self.non_linear(x)

        out = self.block(x)
        residual = self.skip(x)

        out = out + residual
        return out
