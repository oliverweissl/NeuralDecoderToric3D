from ..auxiliary_components import circular_conv_2d, custom_init
from torch import nn, Tensor
from ..residual_blocks import WideResBlock
import torch


class TheEND(nn.Module):
    """The END (https://arxiv.org/abs/2304.07362)."""

    def __init__(
            self,
            channels: list[int],
            depths: list[int],
            lattice_size: int,
            num_classes: int = 16,
            in_channels: int = 2,
            kernel_size: int = 3,
            **kwargs,
    ) -> None:
        """
        Initialize the Model.

        :param num_classes: The number of classes to classify.
        :param kernel_size: The sizes of the kernels used.
        :param lattice_size: The lattice size.
        :param channels: The channel amount wanted.
        :param depths: The depths of individual blocks.
        :param in_channels: The channels in the input.
        """

        super(TheEND, self).__init__()
        self.lattice_size = lattice_size
        self.num_classes = num_classes

        """Initial and Last convolution in the network."""
        self.conv_in = circular_conv_2d(in_channels, channels[0], kernel_size)
        self.conv_out = circular_conv_2d(channels[-1], num_classes, kernel_size, bias=True)

        """The wide-res blocks used in the network."""
        initial_block = self.make_block(channels[0], channels[0], kernel_size, depths[0])

        blocks = [initial_block] + [
            self.make_block(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size,
                depth=depths[0]
            ) for i in range(len(channels)-1)
        ]
        self.blocks = nn.Sequential(*blocks)

        """Auxiliary layers such as normalization and non-linearity."""
        self.batch_norm = nn.BatchNorm2d(channels[-1])
        self.non_linear = nn.GELU()

        """Initialize weights and enable training."""
        self.apply(custom_init)  # Initialize layer weights

    def forward(self, x: Tensor) -> Tensor:
        b, *_ = x.shape
        x = x.reshape(b, 2, self.lattice_size, self.lattice_size)

        """The network body."""
        x = self.conv_in(x)

        x = self.blocks(x)

        x = self.batch_norm(x)
        x = self.non_linear(x)
        x = self.conv_out(x)

        """The network head."""
        x = torch.roll(x, (-1, -1), (2, 3))
        x = torch.permute(x, dims=(0, 2, 3, 1))  # (b, 16, l, l) -> (b, l, l, 16)
        x = torch.flip(x, [1, 2])  # (b, x, y, 16) -> (b, -x, -y, 16)
        x = x.reshape(b, self.lattice_size, self.lattice_size, 4, 2, 2)  # (b, l, l, 16) -> (b, l, l, 4, 2, 2)
        return x

    @staticmethod
    def make_block(in_channels: int, out_channels: int, kernel_size: int, depth: int) -> nn.Sequential:
        """
        Initialize a single block.

        :param in_channels: The number of channels in the input.
        :param out_channels: The number of channels in the output.
        :param kernel_size: The kernel size.
        :param depth: The depth of the block.
        :returns: The blocks as a Sequential layer.
        """
        layers = [
            WideResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                layer_type=circular_conv_2d,
                norm_type=nn.BatchNorm2d,
            ) for _ in range(depth)
        ]
        return nn.Sequential(*layers)