import os
import wandb
import torch
import logging
import hydra
import omegaconf

from torch import nn
from hydra.utils import instantiate
from models import Decoder, MViT
from models.loss_functions import DynamicCELoss
from src import Trainer
from panqec.codes import StabilizerCode


@hydra.main(config_path="config", config_name="3d", version_base="1.2")
def main(args) -> None:
    """
    Start the experiments for decoder training.

    Note that we use hydra for experimental setup and configs, however, it is not required to run the experiments.
    Simply remove the hydra part and set the arguments in the file or use another parser method.
    All arguments needed start with 'args.'.

    :param args: The parsed hydra arguments.
    """
    """Init variables for later use."""
    os.environ["WANDB_API_KEY"]: str = args.default.wandb_api  # parse wandb API key for logging. 
    device = torch.device("cuda:0")

    L: int = args.default.L  # parse lattice size (code is symmetric and as such follows L x L x ..).
    p: float = args.default.p  # parse the error rate [0,1).
    logging.info(f"Lattice size: {L}, Error rate: {p}")

    """Start Wandb experiment."""
    wandb.init(project=args.default.project_name, tags=[str(L), str(p)], entity=args.default.entity)
    wandb_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    wandb.config.update(wandb_args)

    """Initialize the stabilizer Code."""
    code: StabilizerCode = instantiate(args.default.code, L)  # Instantiate the error correcting code using panqec.

    """Make Decoder Model."""
    pooling: nn.Module = instantiate(args.default.pooling, L)  # Instantiate the pooling approach. Pooling layers can be found in 'models/pooling_layers'.
    network: nn.Module = instantiate(args.default.network, **args.net, lattice_size=L)  # Instantiate the network decoder. Decoders can be found in 'models'.
    ensemble = MViT(
        lattice_size=L,
        patch_size=L,
    ) if args.default.network.ensemble else None  # Boolean value determines if the ensemble method is used for decoding.

    decoder = Decoder(network=network, pooling=pooling, ensemble=ensemble)
    decoder.to(device)

    """Instantiate Optimizer, Scheduler and Loss."""
    optimizers, schedulers = [], []

    optimizers.append(opt := torch.optim.AdamW(params=network.parameters(), lr=1e-3, weight_decay=1e-4))
    schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=0.01,
        epochs=args.default.epochs,  # Define the amount of epochs to train (int).
        steps_per_epoch=args.default.batches  # Define the amount of batches per epoch (int).
    ))

    if args.default.network.ensemble:
        optimizers.append(ens_opt := torch.optim.AdamW(params=ensemble.parameters(), lr=1e-3, weight_decay=1e-4))
        schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
            optimizer=ens_opt,
            max_lr=0.01,
            epochs=args.default.epochs,
            steps_per_epoch=args.default.batches
        ))

    criterion = DynamicCELoss(2**(2*code.k), device)

    """Setup Trainer and start training"""
    logging.info("Start Training")

    trainer = Trainer(
        model=decoder,
        loss_function=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        args=args,
        save_model=args.save_model  # Define whether the model should be saved after training.
    )
    """Start training."""
    trainer.train(
        code=code,
        error_rate=p,
    )


if __name__ == '__main__':
    logger = logging.Logger("default_logger")
    logger.setLevel(logging.INFO)
    main()
