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


@hydra.main(config_path="config", config_name="3d", version_base="1.2")
def main(args) -> None:
    """Init some values."""
    os.environ["WANDB_API_KEY"] = args.default.wandb_api
    device = torch.device("cuda:0")

    L = args.default.L
    p = args.default.p
    logging.info(f"Lattice size: {L}, Error rate: {p}")

    """Start Wandb experiment."""
    wandb.init(project=args.default.project_name, tags=[str(L), str(p)], entity=args.default.entity)
    wandb_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    wandb.config.update(wandb_args)

    """Initialize the stabilizer Code."""
    code = instantiate(args.default.code, L)

    """Make Decoder Model."""
    pooling = instantiate(args.default.pooling, L)
    network = instantiate(args.default.network, **args.net, lattice_size=L)
    ensemble = MViT(
        lattice_size=L,
        patch_size=L,
    ) if args.default.network.ensemble else None

    decoder = Decoder(network=network, pooling=pooling, ensemble=ensemble)
    decoder.to(device)

    """Instantiate Optimizer, Scheduler and Loss."""
    optimizers, schedulers = [], []

    optimizers.append(opt := torch.optim.AdamW(params=network.parameters(), lr=1e-3, weight_decay=1e-4))
    schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=0.01,
        epochs=args.default.epochs,
        steps_per_epoch=args.default.batches
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
        save_model=args.save_model
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
