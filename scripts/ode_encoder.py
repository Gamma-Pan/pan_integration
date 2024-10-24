import torch
from torch import nn, exp
import multiprocessing

from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from torchdyn.core import NeuralODE

from pan_integration.models.auto_encoders import *
from pan_integration.data import MNISTDataModule
from pan_integration.utils.callbacks import DigitsReconstruction, LatentSpace

if __name__ == "__main__":
    channels = [16, 32, 64, 64]
    latent_dim = 2
    img_size = [1, 32, 32]
    num_workers = multiprocessing.cpu_count()

    mnist_module = MNISTDataModule(num_workers=num_workers, batch_size=128)
    wandb_logger = WandbLogger(project="ode_encoder")
    callbacks = [
        DigitsReconstruction(num_digits=12),
        LearningRateMonitor(logging_interval="epoch"),
        LatentSpace()
    ]

    # model = AutoEncoder(channels, latent_dim, img_size)
    # model = VarAutoEncoder(channels, latent_dim, img_size)
    # model = ODEAutoEncoderFC(channels, latent_dim, img_size)
    model = ODEAutoEncoderConv(channels, 32,latent_dim , img_size)

    trainer = Trainer(
        fast_dev_run=False,
        max_epochs=50,
        enable_checkpointing=False,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=mnist_module)
