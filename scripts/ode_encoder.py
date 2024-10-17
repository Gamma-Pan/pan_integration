import torch
from torch import nn, exp
import multiprocessing

from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.cpu.amp import autocast

from torchdyn.core import NeuralODE

from pan_integration.models import VariationalAutoEncoder
from pan_integration.models.auto_encoders import *
from pan_integration.data import MNISTDataModule
from pan_integration.utils.callbacks import AutoEncoderViz

if __name__ == "__main__":
    channels = [8, 8, 16, 16 ]
    latent_dim = 5
    img_size = [1, 32, 32]
    num_workers = multiprocessing.cpu_count()

    mnist_module = MNISTDataModule(num_workers=num_workers)
    wandb_logger = WandbLogger(project="ode_encoder")
    callbacks = [
        AutoEncoderViz(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # autoenc = AutoEncoder(channels, latent_dim, img_size)
    # autoenc = ODEAutoEncoder(channels, latent_dim, img_size)
    # autoenc = VariationalAutoEncoder(channels, latent_dim, img_size)
    autoenc = VEC(channels, latent_dim, img_size)

    trainer = Trainer(
        fast_dev_run=False,
        max_epochs=50,
        enable_checkpointing=False,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(autoenc, datamodule=mnist_module)
