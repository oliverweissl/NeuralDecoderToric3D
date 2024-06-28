import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb
from src.metrics import WandbMetrics
from typing import Callable, Type
from ._data_generator import DataGenerator
from panqec.codes import StabilizerCode
import logging
import time


class Trainer:
    """A trainer that generates batches on the fly."""
    model: nn.Module
    optimizers: list[Optimizer]
    schedulers: list[LRScheduler]
    evaluator: Callable
    criterion: nn.Module
    training_samples: int

    _output: Type[Callable]  # An output function to either print or log progress.

    """Parameters for the training."""
    _batch_size: int
    _num_epochs: int
    _num_batches: int

    """Variables for saving models."""
    _save_model: bool
    _save_directory: str

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            optimizers: list[Optimizer],
            schedulers: list[LRScheduler],
            args,
            verbose: bool = False,
            save_model: bool = False,
    ) -> None:
        """
        Initialize the trainer object.

        :param model: The decoder model.
        :param loss_function: The Loss function.
        :param optimizers: The optimizer.
        :param schedulers: The scheduler.
        :param args: Arguments for the Trainer.
        :param verbose: Whether the trainer should print progress or log it.
        :param save_model: If model should be saved.
        :return: The trained decoder and train / validation values.
        """
        self.model = model

        self._output = print if verbose else logging.info
        self.criterion = loss_function
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.scaler = torch.cuda.amp.GradScaler()

        self._num_batches = args.default.batches
        self._num_epochs = args.default.epochs
        self._batch_size = args.batch_size
        self._save_directory = wandb.run.dir
        self._save_model = save_model

    def train(
            self, *,
            code: StabilizerCode,
            error_rate: float
    ) -> None:
        """
        Train the neural decoder on dynamically generated data.

        :param code: The code to train the decoder on.
        :param error_rate: The error rate to train the decoder on.
        """
        """Extract parameters."""
        torch.backends.cudnn.benchmark = True  # Enable cuda to find the best tuner for hardware.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_generator = DataGenerator(code=code, verbose=False, error_rate=error_rate, batch_size=self._batch_size)

        """Start Training."""
        for epoch in range(self._num_epochs):
            self._output(f"{'=' * 18}")
            self._output(f"Starting Epoch {epoch}.")
            epoch_start = time.time()

            """Train Model"""
            loss, _ = self._process_batches(data_generator, device, self._num_batches)
            epoch_time = time.time() - epoch_start

            """Evaluate model."""
            self._output("Evaluating Model.")
            with torch.no_grad():
                _, (y_pred, y_true) = self._process_batches(data_generator, device, 1, train=False)

            """Record evaluation Metrics."""
            metrics = WandbMetrics.get_metrics(
                y_pred=y_pred,
                y_true=y_true,
                loss=loss,
                learning_rate=self.schedulers[0].optimizer.param_groups[0]['lr'],
                epoch_duration=epoch_time,
            )
            wandb.log(metrics.__dict__)

        """Sve the finished model."""
        if self._save_model:
            self._output("Saving Model.")
            self.save_model(path=self._save_directory, model_name=wandb.run.name)

    def _process_batches(
            self,
            data_generator: DataGenerator,
            device: torch.device,
            batches: int,
            train: bool = True,
    ) -> tuple[float, tuple[Tensor, Tensor]]:
        """
        Process epoch and log if it is testing.
        
        :param data_generator: The data generator object.
        :param device: The device to run the loop on.
        :param batches: The amount of batches to train.
        :param train: Whether its training or not.
        :returns: The loss and a tuple of (y_pred, y_true).
        :raises ValueError: If loss is nan.
        """

        loss = 0.
        for _ in range(batches):
            X, y = data_generator.generate_batch(use_qmc=train, device=device)
            """Zero out the gradient for all optimizers."""
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            """Forward pass."""
            with torch.autocast("cuda"):
                y_pred = self.model(X)
                loss_c = self.criterion(y_pred, y)

            if train:
                """Record loss."""
                loss += loss_c.item()

                """Backward pass."""
                self.scaler.scale(loss_c).backward()

                """Update weights and step schedulers."""
                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)

                self.scaler.update()
                for scheduler in self.schedulers:
                    scheduler.step()
        return loss / batches, (y_pred, y)

    def save_model(self, path: str = ".", model_name: str = "model") -> None:
        """
        Save the current model in the Trainer.

       :param path: The path to save it to.
       :param model_name: The name of the saved model.
        """
        torch.save(self.model, f"{path}/{model_name}.pt")
