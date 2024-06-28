import torch
from torch import nn, Tensor
from torch.functional import F


class DynamicCELoss(nn.Module):
    """A BCE Loss adaptation, that dynamically adapts class weights."""

    def __init__(self, tensor_size: int, device: torch.device) -> None:
        """
        Initialize the DynamicBCELoss.

        :param tensor_size: The size of tensor to accept.
        :param device: The device to use the loss on.
        """
        super(DynamicCELoss, self).__init__()
        self.logit_counter = torch.zeros(size=(tensor_size,), device=device) + 1e-6 # This counts the number of occurrences for certain logicals.
        self.global_counter = torch.zeros(size=(tensor_size,), device=device)  # This counts the total number of samples viewed.
        self.num_classes = tensor_size

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        The forward pass of the loss function.

        :param output: The output tensor of the NN in form of (b, tensor_size).
        :param target: The target tensor in the same form.
        """
        """Calculating weights and updating counters."""
        self.logit_counter = self.logit_counter + target.sum(dim=0)
        self.global_counter = self.global_counter + target.shape[0]
        weights = (self.global_counter / (self.logit_counter * self.num_classes))

        """Calculating loss."""
        loss = F.cross_entropy(output, target, weights)
        return loss
