from torch import nn


def circular_conv_2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = False) -> nn.Conv2d:
    """
    Return a circular padded 2d convolution layer.

    :param in_channels: The number of channels in the input.
    :param out_channels: The number of channels in the output.
    :param kernel_size: The kernel size.
    :param bias: Whether bias is used.
    :returns: The convolution layer, with kaiming initialized weights.
    """
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", padding_mode="circular", bias=bias)
    nn.init.kaiming_normal_(convolution.weight)
    return convolution
