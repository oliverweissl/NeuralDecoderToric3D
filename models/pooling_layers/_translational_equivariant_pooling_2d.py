from torch import nn, Tensor
import torch


class TranslationalEquivariantPooling2D(nn.Module):
    """Translational Equivariant Pooling Layer."""

    def __init__(self, l: int) -> None:
        """Initialize the pooling layer."""
        self.l = l
        super(TranslationalEquivariantPooling2D, self).__init__()

    def get_logic_action_flag(self, syndrome_string: Tensor, axis: int, shift: int) -> Tensor:
        """
        Get the conditional array for a syndrome on either the dual or primal lattice.

        :param syndrome_string: The syndrome.
        :param axis: The axis to operate on.
        :param shift: The shift.
        :returns: The condition array.
        """
        b, *_ = syndrome_string.shape

        """Translate syndrome into Square."""
        syndrome = syndrome_string.reshape(b, self.l, self.l)
        rolled_syndrome = torch.roll(syndrome, shifts=shift, dims=2 - axis)  # Axis shifted by one.

        """Make Syndrome into Vector."""
        rolled_syndrome = torch.sum(rolled_syndrome, dim=1 + axis)
        rolled_syndrome = torch.flip(rolled_syndrome, dims=(-1,))
        rolled_syndrome = torch.roll(rolled_syndrome, shifts=1, dims=-1)  # Axis shifted by zero.

        """Convert vector to conditions."""
        condition_array = torch.cumsum(rolled_syndrome, dim=-1)
        condition_array = torch.roll(condition_array, shifts=1, dims=-1)
        condition_array = (condition_array % 2) > 0
        return condition_array

    def logic_action_average(self, x: Tensor, syndrome: Tensor, axis: int) -> Tensor:
        """
        Get the probabilities adjusted by the condition matrix.

        :param x: The probability tensor (b, L, L, 4, 2, 2).
        :param syndrome: The syndrome string.
        :param axis: The axis of the action (x, y) -> (0,1).
        :returns: The adjusted Tensor (b, L, L, 4, 2, 2).
        """
        b, l, _, *tail = x.shape
        primal_syndrome, dual_syndrome = torch.hsplit(syndrome, 2)

        flags_primal = self.get_logic_action_flag(primal_syndrome, shift=1, axis=axis)
        flags_dual = self.get_logic_action_flag(dual_syndrome, shift=0, axis=axis)

        primal_cond_repeated = flags_primal.unsqueeze(-1).repeat(1, 1, l).transpose(1, 2)  # (b, l) -> (b, l, l)
        dual_cond_repeated = flags_dual.unsqueeze(-1).repeat(1, 1, l).transpose(1, 2)

        """Apply modification for all non identity actions on primal."""
        x[primal_cond_repeated, ...] = torch.roll(x[primal_cond_repeated, ...], shifts=1, dims=-1 - axis)

        """Apply modification for all non identity actions on dual. -> We flip cross and squares."""
        candidate = x[dual_cond_repeated, ...]
        bc, *_ = candidate.shape
        commutation = [0, 2, 1, 3]

        candidate = candidate.permute(0, 3, 2, 1).reshape(bc, *tail)[:, commutation, ...]
        candidate = torch.roll(candidate, shifts=1, dims=-2 + axis)
        x[dual_cond_repeated, ...] = candidate.permute(0, 3, 2, 1).reshape(bc, *tail)[:, commutation, ...]
        return x

    def forward(self, x: Tensor, syndrome: Tensor) -> Tensor:
        for i in range(2):
            x = self.logic_action_average(x, syndrome, axis=i)
            x = x.transpose(1, 2)

        x = torch.mean(x, dim=(1, 2))
        return x
