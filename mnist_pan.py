import matplotlib.pyplot as plt
import torch
from torch import nn, tensor
from torch.nn import functional as F

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler

from torchdyn.core import MultipleShootingLayer, NeuralODE

from pan_integration.data import MNISTDataModule
from pan_integration.core.ode import PanODE, PanZero
from pan_integration.utils.lightning import LitOdeClassifier

import wandb

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

import multiprocessing as mp

NUM_WORKERS = mp.cpu_count()
CHANNELS = 64

embedding = nn.Sequential(
    nn.Conv2d(1, 64, 5, 2),
    nn.GroupNorm(4, CHANNELS),
    nn.Tanh(),
).to(device)


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0
        self.norm1 = nn.GroupNorm(4, CHANNELS)
        self.norm2 = nn.GroupNorm(4, CHANNELS)
        self.norm3 = nn.GroupNorm(4, CHANNELS)
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1)
        self.nonlin = nn.ReLU()

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = self.nonlin(self.norm1(self.conv1(x)))
        x = self.nonlin(self.norm2(self.conv2(x)))
        x = self.nonlin(self.norm3(self.conv3(x)))
        return x


classifier = nn.Sequential(
    nn.Conv2d(CHANNELS, 1,1,1),
    nn.Flatten(),
    nn.Dropout(0.1),
    nn.Linear(12*12, 10),
)


def train_mnist_ode(t_span, ode_model, epochs=10, test=False, loggers=()):
    learner = LitOdeClassifier(embedding, ode_model, classifier)
    dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    trainer = Trainer(
        max_epochs=epochs,
        enable_checkpointing=False,
        fast_dev_run=False,
        accelerator="gpu",
        logger=loggers,
    )

    trainer.fit(learner, datamodule=dmodule)
    if test:
        trainer.test(learner, datamodule=dmodule)


if __name__ == "__main__":
    t_span = torch.linspace(0, 1, 2).to(device)
    vf = VF().to(device)
    num_coeff_per_dim = num_points = 32

    solver = PanZero(
        num_coeff_per_dim,
        num_points,
        delta=1e-3,
        max_iters=30,
        device=device,
    )
    solver_adjoint = PanZero(
        num_coeff_per_dim,
        num_points,
        delta=1e-3,
        max_iters=30,
        device=device,
    )
    ode_model = PanODE(vf, t_span, solver, solver_adjoint)
    # ode_model = NeuralODE(VF().to(device), solver="tsit")

    logger = WandbLogger(project='pan_integration')
    logger.experiment.config['name'] = 'pan_32_32'
    logger.experiment.config["num_coeff_per_dim"] = num_coeff_per_dim
    logger.experiment.config["num_points"] = num_points
    train_mnist_ode(t_span, ode_model, epochs=30, test=True, loggers=(logger,))
    wandb.finish()
