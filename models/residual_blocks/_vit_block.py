from torch import nn, Tensor
from typing import Callable


class ViTBlock(nn.Module):
    """Visual Transformer Residual Block."""

    def __init__(self, hidden_d: int, n_heads: int, attention: Callable, mlp_ratio=4) -> None:
        """
        Initialize the block.

        :param hidden_d: The hidden dimensionality.
        :param mlp_ratio: The ratio of expansion for the MLP.
        """
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_d)
        self.multi_head_self_attention = attention(hidden_d, n_heads)

        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)
        x = self.multi_head_self_attention(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x
