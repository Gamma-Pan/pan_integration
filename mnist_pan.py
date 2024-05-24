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
CHANNELS = 32
NUM_GROUPS = 4
WANDB_LOG = True

embedding = nn.Sequential(
    # nn.Conv2d(1, CHANNELS, 3, 2, padding=1),
    # nn.GroupNorm(NUM_GROUPS, CHANNELS),
    # nn.Tanh(),
    nn.Flatten()
).to(device)


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0
        # self.norm1 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        # self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1)
        # self.norm2 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        # self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1)
        self.lin1 = nn.Linear(28**2, 28**2)
        self.lin2 = nn.Linear(28**2, 28**2)
        self.lin3 = nn.Linear(28**2, 28**2)
        self.nl_fn = nn.ReLU()

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        # x = self.nl_fn(self.norm1(self.conv1(x)))
        # x = self.nl_fn(self.norm2(self.conv2(x)))
        x = self.nl_fn(self.lin1(x))
        x = self.nl_fn(self.lin2(x))
        x = self.nl_fn(self.lin3(x))
        return x


classifier = nn.Sequential(
    # nn.Dropout(0.01),
    # nn.Conv2d(CHANNELS, 1, 3, 1),
    # nn.ReLU(),
    # nn.Flatten(),
    # nn.Linear(144, 10),
    nn.Linear(28**2, 10),
)


def train_mnist_ode(t_span, ode_model, epochs=10, test=False, logger=()):
    learner = LitOdeClassifier(t_span, embedding, ode_model, classifier)
    dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    trainer = Trainer(
        max_epochs=epochs,
        enable_checkpointing=False,
        fast_dev_run=False,
        accelerator="gpu",
        logger=logger,
    )

    trainer.fit(learner, datamodule=dmodule)
    if test:
        trainer.test(learner, datamodule=dmodule)


def train_all_pan(configs=None, sensitivity='autograd', epochs=50):
    t_span = torch.linspace(0,1,10).to(device)

    if configs is None:
        configs = (
            {"num_coeff_per_dim": 16, "num_points": 16, "delta": 1e-3, "max_iters": 20},
            {"num_coeff_per_dim": 32, "num_points": 32, "delta": 1e-3, "max_iters": 20},
            {"num_coeff_per_dim": 64, "num_points": 64, "delta": 1e-3, "max_iters": 20},
            {"num_coeff_per_dim": 16, "num_points": 16, "delta": 1e-2, "max_iters": 20},
            {"num_coeff_per_dim": 32, "num_points": 32, "delta": 1e-2, "max_iters": 20},
            {"num_coeff_per_dim": 64, "num_points": 64, "delta": 1e-2, "max_iters": 20},
        )

    for config in configs:
        vf = VF().to(device)
        logger=()
        if WANDB_LOG:
            logger = WandbLogger(
                project="pan_integration",
                name=f"pan_{config.num_coeff_per_dim}_{config.num_points}",
                log_model=False,
            )
            logger.experiment.config.update(config)
            logger.experiment.config.update({"type": "pan", "sensitivity": sensitivity})

        model = PanODE(vf, t_span, solver=config, solver_adjoint=config, sensitivity=sensitivity).to(device)
        train_mnist_ode(t_span, model, epochs=epochs, test=True, logger=logger)


def train_all_shooting(configs=None, sensitivity='autograd', epochs=50):
    t_span = torch.linspace(0,1,10).to(device)

    if configs is None:
        configs = (
            {"solver":'dopri5', "atol":1e-3 },
            {"solver":'tsit5', "atol":1e-3 ,},
            {"solver":'dopri5', "atol":1e-4 },
            {"solver":'tsit5', "atol":1e-4 },
            {"solver":'rk-4',}
        )

    for config in configs:
        vf = VF().to(device)
        logger = ()
        if WANDB_LOG:
            logger = WandbLogger(
                project="pan_integration",
                name=f"shooting_f{config.solver}",
                log_model=False,
            )
            logger.experiment.config.update(config)
            logger.experiment.config.update({"type": "shooting", "sensitivity": sensitivity, 'fixed_step':10})

        model = NeuralODE(vf,  **config, sensitivity=sensitivity)
        train_mnist_ode(t_span, model, epochs=epochs, test=True, logger=logger)


if __name__ == "__main__":
    train_all_pan(epochs=50)
    train_all_shooting(epochs=50)



