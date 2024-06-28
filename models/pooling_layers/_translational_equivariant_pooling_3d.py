from torch import nn, Tensor
import torch


class TranslationalEquivariantPooling3D(nn.Module):
    """Translational Equivariant Pooling Layer."""

    def __init__(self, l: int) -> None:
        """Initialize the pooling layer."""
        self.l = l
        self.permute_order = (0, 2, 3, 1, 4, 5, 6, 7)  # (b, x, y, z, 8, 2, 2, 2) -> (b, y, z, x, 8, 2, 2, 2)
        super(TranslationalEquivariantPooling3D, self).__init__()

    def get_z_flag(self, syndrome_string: Tensor, axis: int, shift: int) -> Tensor:
        """
        Get the conditional array for a syndrome on the dual lattice.

        :param syndrome_string: The syndrome.
        :param axis: The axis to operate on.
        :param shift: The shift.
        :returns: The condition array.
        """
        b, *_ = syndrome_string.shape

        """Translate syndrome into Cube."""
        syndrome = syndrome_string.reshape(b, self.l, self.l, self.l)
        rolled_syndrome = torch.roll(syndrome, shifts=shift, dims=3 - axis)  # Axis shifted by one.

        """Make Syndrome into Surface."""
        rolled_syndrome = torch.sum(rolled_syndrome, dim=1 + axis)
        rolled_syndrome = torch.flip(rolled_syndrome, dims=(-1,))
        rolled_syndrome = torch.roll(rolled_syndrome, shifts=1, dims=-1)  # Axis shifted by zero.

        """Convert surface to conditions."""
        condition_array = torch.cumsum(rolled_syndrome, dim=-1)
        condition_array = torch.roll(condition_array, shifts=1, dims=-1)
        condition_array = (condition_array % 2) > 0
        return condition_array

    def get_x_flag(self, syndrome_string: Tensor, axis: int, shift: int) -> Tensor:
        """
        Get the conditional array for a syndrome on the primal lattice.

        :param syndrome_string: The syndrome.
        :param axis: The axis to operate on.
        :param shift: The shift.
        :returns: The condition array.
        """
        b, *_ = syndrome_string.shape

        """Translate syndrome into Cube."""
        syndrome = syndrome_string.reshape(b, self.l, self.l, self.l)
        rolled_syndrome = torch.roll(syndrome, shifts=shift, dims=3 - axis)  # Axis shifted by one.

        """Make Syndrome into Surface."""
        rolled_syndrome = torch.sum(rolled_syndrome, dim=(1 + axis, 1 + (1 + axis) % 3))
        rolled_syndrome = torch.flip(rolled_syndrome, dims=(-1,))
        rolled_syndrome = torch.roll(rolled_syndrome, shifts=1, dims=-1)  # Axis shifted by zero.

        """Convert surface to conditions."""
        condition_array = torch.cumsum(rolled_syndrome, dim=-1)
        condition_array = torch.roll(condition_array, shifts=1, dims=-1)
        condition_array = (condition_array % 2) > 0
        return condition_array

    def logic_action_average(self, probs: Tensor, syndrome: Tensor, axis: int) -> Tensor:
        """
        Get the probabilities adjusted by the condition matrix.

        :param probs: The probability tensor (b, L, L, L, 8, 2, 2, 2).
        :param syndrome: The syndrome string.
        :param axis: The axis of the action (x, y, z) -> (0,1,2).
        :returns: The adjusted Tensor (b, L, L, L, 8, 2, 2, 2).
        """
        b, l, *_ = probs.shape
        z_syndrome, *x_syndrome_channels = torch.hsplit(syndrome, 4)

        """Get Z Flags."""
        flags_z = self.get_z_flag(z_syndrome, shift=0, axis=axis)
        flags_z = flags_z.unsqueeze(-1).repeat(1, 1, 1, l).permute(0, 3, 2, 1)  # (b, l, l) -> (b, l, l, l)

        """Apply modification for all non identity actions on dual. -> We flip cross and squares."""
        z_candidate = probs[flags_z, ...]
        shape = z_candidate.shape
        z_candidate = self._swap_stabilizers(z_candidate, shape)
        z_candidate = torch.roll(z_candidate, shifts=1, dims=-3 + axis)

        probs[flags_z, ...] = self._swap_stabilizers(z_candidate, shape)

        """Get Primal Flags."""
        channel = x_syndrome_channels[axis]
        flags_x = self.get_x_flag(channel, shift=1, axis=axis)
        flags_x = flags_x[..., None, None].repeat(1, 1, l, l).permute(0, 3, 2, 1)

        """Apply modification for all non identity actions on primal."""
        probs[flags_x, ...] = torch.roll(probs[flags_x, ...], shifts=1, dims=-1 - axis)
        return probs

    @staticmethod
    def _swap_stabilizers(tensor: Tensor, target_shape: tuple[int, ...]) -> Tensor:
        """
        Swap the cube stabilizers to be cross stabilizers and vice-versa.

        :param tensor: The tensor.
        :param target_shape: The original shape of the tensor.
        :returns: the switched tensor.
        """
        tensor = tensor.permute(0, 4, 3, 2, 1)
        tensor = tensor.reshape(target_shape)
        tensor = tensor[:, [0, 6, 2, 4, 3, 5, 1, 7], ...]  # Commutation relations
        return tensor

    def forward(self, x: Tensor, syndrome: Tensor) -> Tensor:
        for i in range(3):
            x = self.logic_action_average(x, syndrome, axis=i)
            x = x.permute(*self.permute_order)

        x = torch.mean(x, dim=(1, 2, 3))
        return x
