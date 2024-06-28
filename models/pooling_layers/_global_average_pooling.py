from torch import nn, Tensor


class GlobalAveragePooling(nn.Module):
    """Traditional global average pooling layer."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize this layer."""
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x: Tensor, *_) -> Tensor:
        """
        The forward pass for the Global Average Pooling.

        :param x: The output of the NN (b, (l,) *d, n, (2,)*d).
        :return: The output.
        """
        x = x.mean(dim=(1, 2, 3))
        return x
