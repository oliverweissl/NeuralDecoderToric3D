import torch
from torch import nn
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .auxiliary_components import circular_conv_3d


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.non_linear = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.liner_in = nn.Linear(dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, dim)

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        x = self.norm(x)
        x = self.liner_in(x)
        x = self.non_linear(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = math.sqrt(dim_head)

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)

        self.linear_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.linear_out = nn.Linear(inner_dim, dim, bias=False) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.linear_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q / self.scale

        attention_score = torch.matmul(q, k.transpose(-1, -2))
        attention = self.attend(attention_score )
        out = torch.matmul(attention, v)

        b, h, c, d = out.shape
        out = torch.reshape(out, (b, c, h * d))
        out = self.linear_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                MLP(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MViT(nn.Module):
    """Borrowed from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py."""

    def __init__(
            self, *,
            lattice_size: int,
            patch_size: int,
            dim: int = 256,
            depth: int = 4,
            heads: int = 8,
            mlp_dim: int = 256,
            in_channels: int = 4,
            num_classes: int = 64,
            dim_head: int = 64,
    ):
        super().__init__()
        self.lattice_size = lattice_size
        assert lattice_size % patch_size == 0, 'Lattice size must be divisible by the patch size.'

        num_patches = (lattice_size // patch_size) ** 3
        patch_dim = in_channels * (patch_size ** 3)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f p1) (h p2) (w p3) -> b (f h w) (p1 p2 p3 c)', p1=patch_size, p2=patch_size,
                      p3=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.initial_conv = circular_conv_3d(in_channels, in_channels, lattice_size)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        """The head of the ViT"""
        self.layer_norm = nn.LayerNorm(dim)
        self.fully_connected = nn.Linear(dim, num_classes)

    def forward(self, x):
        b, *_ = x.shape
        x = x.reshape(b, 4, self.lattice_size, self.lattice_size, self.lattice_size)
        x = self.initial_conv(x)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)

        x = x[:, 0]  # This is class-pooling, can be switched to mean pooling: x.mean(dim=1)
        x = self.layer_norm(x)
        x = self.fully_connected(x)
        return x
