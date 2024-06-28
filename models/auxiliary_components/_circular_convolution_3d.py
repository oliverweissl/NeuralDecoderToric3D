from torch import nn


def circular_conv_3d(in_channels: int, out_channels: int, kernel_size: int, bias=False) -> nn.Conv3d:
    """
    Return a circular padded 3d convolution layer.

    :param in_channels: The number of channels in the input.
    :param out_channels: The number of channels in the output.
    :param kernel_size: The kernel size.
    :param bias: Whether bias is used.
    :returns: The convolution layer.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding="same", padding_mode="circular", bias=bias)
