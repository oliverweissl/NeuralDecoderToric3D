import torch
from torch import nn, Tensor
from .auxiliary_components import circular_conv_3d, custom_init, AConvCircular3D
from .residual_blocks import WideResBlock


class TransformedEND(nn.Module):
    """Base model inpired by ResNet and WideResNet + the END."""

    def __init__(
            self,
            channels: list[int],
            depths: list[int],
            lattice_size: int,
            num_classes: int = 64,
            in_channels: int = 4,
            kernel_size: int = 3,
            **kwargs,
    ) -> None:
        """
        Initialize the Model.

        :param num_classes: The number of classes to classify.
        :param kernel_size: The sizes of the kernels used.
        :param lattice_size: The size of the lattice.
        :param channels: The channel amount wanted.
        :param depths: The depths of individual blocks.
        :param in_channels: The channels in the input (default of 1 due to toric code dim).
        """
        super(TransformedEND, self).__init__()
        self.lattice_size = lattice_size

        """Initial convolution in the network."""
        self.conv_in = AConvCircular3D(in_channels, channels[0], kernel_size, attention_channels=5, number_heads=5,
                                       key_depths=5 * 5)
        self.conv_out = AConvCircular3D(channels[-1], num_classes, kernel_size, attention_channels=5, number_heads=5,
                                        key_depths=5 * 5, bias=True)

        """The wide-res blocks used in the network."""
        initial_block = self.make_block(channels[0], channels[0], kernel_size, depths[0])

        blocks = [initial_block] + [
            self.make_block(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size,
                depth=depths[i + 1]
            ) for i in range(len(channels) - 1)
        ]
        self.blocks = nn.Sequential(*blocks)

        """Auxiliary layers sucha s normalization, pooling and the final output processing."""
        self.batch_norm = nn.BatchNorm3d(channels[-1])
        self.non_linear = nn.GELU()

        self.apply(custom_init)  # Initialize layer weights

    @torch.autocast(device_type="cuda")
    def forward(self, x: Tensor) -> Tensor:
        b, *_ = x.shape
        x = x.reshape(b, 4, self.lattice_size, self.lattice_size, self.lattice_size)

        """The network body."""
        x = self.conv_in(x)
        x = self.blocks(x)

        x = self.batch_norm(x)
        x = self.non_linear(x)
        x = self.conv_out(x)

        """The networks head."""
        x = torch.roll(x, (-1, -1, -1), (2, 3, 4))
        x = torch.permute(x, dims=(0, 2, 3, 4, 1))  # (b, 64, l, l, l) -> (b, l, l, l, 64)
        x = torch.flip(x, [1, 2, 3])
        x = x.reshape(b, self.lattice_size, self.lattice_size, self.lattice_size, 8, 2, 2, 2)
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
        inital_layer = WideResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            layer_type=circular_conv_3d,
            norm_type=nn.BatchNorm3d,
        )
        layers = [inital_layer] + [
            WideResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                layer_type=circular_conv_3d,
                norm_type=nn.BatchNorm3d,
            ) for _ in range(depth - 1)
        ]
        return nn.Sequential(*layers)
