from torch import nn, Tensor
from math import prod
from typing import Optional


class Decoder(nn.Module):
    """A decoder that combines a NeuralNetwork with an equivariant pooling layer."""

    def __init__(self, network: nn.Module, pooling: nn.Module, ensemble: Optional[nn.Module]) -> None:
        """
        Initialize the decoder.

        :param network: The neural network.
        :param pooling: The last pooling layer.
        :param ensemble: An additional network to be used for ensemble classification.
        """
        super().__init__()
        self.pooling = pooling
        self.network = network
        self.ensemble = ensemble

        self.network.float()  # Set dtype in network.
        self.network.train()  # Make the network trainable.

        if self.ensemble is not None:
            self.ensemble.float()
            self.ensemble.train()

    def forward(self, syndrome: Tensor) -> Tensor:
        out = self.network(syndrome)  # -> (b, (l,)*d, n=2**d, (2,)*d)  n: code subspace
        out = self.pooling(out, syndrome)  # -> (b, n, (2,)*d)

        b, *tail = out.shape
        out = out.reshape(b, prod(tail))  # -> (b, n*(2**d))
        ens_out = self.ensemble(syndrome) if self.ensemble is not None else out.clone()

        out = (out + ens_out) / 2
        return out
