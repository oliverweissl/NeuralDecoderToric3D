import torch
from torch import nn, Tensor


class GradientInputLayer(nn.Module):
    L: int
    N: int
    bin_size: int

    def __init__(self, L: int, N: int, num_channels: int) -> None:
        """
        Initialize this layer.

        :param L: The lattice size.
        :param N: The gradient size.
        """
        super().__init__()
        self.L = L
        self.N = N
        self.bin_size = 360 // N
        self.norm = nn.BatchNorm2d(num_channels)
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Generate gradients from the syndrome.

        :param x: The syndrome in BSF.
        :returns: The gradient in form (b,4,N,N).
        """
        b, *_ = x.shape
        x = x.reshape(b, self.num_channels, self.L, self.L, self.L)
        gradients = torch.zeros((b, self.num_channels, self.N, self.N), device=x.device)

        iii = torch.nonzero(x).T
        indices = iii[:2]
        mags = torch.sqrt(torch.sum(torch.mul(iii[-3:], iii[-3:]), dim=0))

        x_comp = torch.sqrt(iii[-2] ** 2 + iii[-1] ** 2)
        z_comp = torch.sqrt(iii[-2] ** 2 + iii[-3] ** 2)

        ax = ((torch.atan2(x_comp, iii[-3]) * 180 / torch.pi) / self.bin_size).int()
        az = ((torch.atan2(iii[-1], z_comp) * 180 / torch.pi) / self.bin_size).int()

        targets = torch.cat((indices, ax.unsqueeze(0), az.unsqueeze(0)), dim=0)
        gradients[tuple(targets)] = gradients[tuple(targets)] + mags
        gradients = self.norm(gradients)
        return gradients
