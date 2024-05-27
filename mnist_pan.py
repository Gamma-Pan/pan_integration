import matplotlib.pyplot as plt
import torch
from torch import nn, tensor
from torch.nn import functional as F

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler

from torchdyn.core import MultipleShootingLayer, NeuralODE

from pan_integration.data import MNISTDataModule
from pan_integration.core.ode import PanODE, PanZero
from pan_integration.utils.lightning import (
    LitOdeClassifier,
    NfeMetrics,
    ProfilerCallback,
)

import wandb

from copy import copy
import argparse
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 6

import multiprocessing as mp

NUM_WORKERS = mp.cpu_count()
CHANNELS = 32
NUM_GROUPS = 2
WANDB_LOG = False


class Augmenter(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv = nn.Conv2d(1, CHANNELS - 1, 3, 1, 1)
        self.norm = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.conv2 = nn.Conv2d(CHANNELS - 1, CHANNELS - 1, 3, 2, 1)

    def forward(self, x):
        aug = F.tanh(self.conv(x))
        x = torch.cat([x, aug], dim=1)
        return x


embedding = Augmenter(CHANNELS)


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0
        self.norm1 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.conv3 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, padding=1, bias=False)
        self.norm4 = nn.GroupNorm(NUM_GROUPS, CHANNELS)

    def forward(self, t, x):
        self.nfe += 1
        x = F.relu(self.norm1(x))
        x = F.relu(self.norm2(self.conv1(x)))
        x = F.relu(self.norm3(self.conv2(x)))
        x = self.norm4(self.conv3(x))
        return x


classifier = nn.Sequential(
    nn.Dropout(0.01),
    nn.Conv2d(CHANNELS, 1, 3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(28**2, 10),
)


def train_mnist_ode(t_span, ode_model, epochs=10, test=False, logger=()):
    learner = LitOdeClassifier(t_span, embedding, ode_model, classifier)
    dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    nfe_callback = NfeMetrics()
    # early_callback = EarlyStopping(
    #     monitor="val_accuracy",
    #     check_on_train_epoch_end=True,
    #     stopping_threshold=0.91,
    # )

    checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc",
        mode="min",
    )

    prof_callback = ProfilerCallback()

    trainer = Trainer(
        max_epochs=epochs,
        enable_checkpointing=True,
        fast_dev_run=False,
        accelerator="gpu",
        logger=logger,
        callbacks=[nfe_callback, checkpoint] # prof_callback],
    )

    trainer.fit(learner, datamodule=dmodule)
    if test:
        trainer.test(learner, datamodule=dmodule)


def train_all_pan(configs, sensitivity, epochs, test):
    t_span = torch.linspace(0, 1, 2).to(device)

    for config in configs:
        vf = VF().to(device)
        logger = ()
        name = f"pan_{config['num_coeff_per_dim']}_{config['num_points']}_{str(config['delta'])}"

        if WANDB_LOG:
            logger = WandbLogger(
                project="pan_integration",
                name=name,
                log_model=False,
            )
            logger.experiment.config.update(config)
            logger.experiment.config.update(
                {
                    "type": "pan",
                    "sensitivity": sensitivity,
                    "architecture": "CNN_IL_aug",
                }
            )

        model = PanODE(
            vf, t_span, solver=config, solver_adjoint=config, sensitivity=sensitivity
        ).to(device)
        train_mnist_ode(t_span, model, epochs=epochs, test=test, logger=logger)
        if WANDB_LOG:
            # profile_artf = wandb.Artifact(f"trace_{name}", type="profile")
            # profile_artf.add_file(local_path="./trace.json")
            # logger.experiment.log_artifact(profile_artf)
            wandb.finish()


def train_all_shooting(configs, sensitivity, epochs, test):
    for config in configs:
        vf = VF().to(device)
        logger = ()
        name = f"shooting_{config['solver']}_{config['atol'] if 'atol' in config.keys() else config['fixed_steps'] }"
        if WANDB_LOG:
            logger = WandbLogger(
                project="pan_integration",
                name=name,
                log_model=False,
            )
            logger.experiment.config.update(config)
            logger.experiment.config.update(
                {
                    "type": "shooting",
                    "sensitivity": sensitivity,
                    "architecture": "CNN_IL_aug",
                }
            )

        _config = copy(config)
        del _config["fixed_steps"]

        model = NeuralODE(vf, **_config, sensitivity=sensitivity)
        t_span = torch.linspace(0, 1, int(config["fixed_steps"])).to(device)
        train_mnist_ode(t_span, model, epochs=epochs, test=test, logger=logger)
        if WANDB_LOG:
            # profile_artf = wandb.Artifact(f"trace_{name}", type="profile")
            # profile_artf.add_file(local_path="./trace.json")
            # logger.experiment.log_artifact(profile_artf)
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--log", default=False, type=bool)
    args = vars(parser.parse_args())
    WANDB_LOG = args["log"]

    pan_configs = (
        # {"num_coeff_per_dim": 16, "num_points": 16, "delta": 1e-3, "max_iters": 10},
        {"num_coeff_per_dim": 32, "num_points": 32, "delta": 1e-3, "max_iters": 20},
        # {"num_coeff_per_dim": 64, "num_points": 64, "delta": 1e-3, "max_iters": 30},
        # {"num_coeff_per_dim": 16, "num_points": 16, "delta": 1e-2, "max_iters": 20},
        # {"num_coeff_per_dim": 32, "num_points": 32, "delta": 1e-2, "max_iters": 20},
        # {"num_coeff_per_dim": 64, "num_points": 64, "delta": 1e-2, "max_iters": 20},
    )

    shoot_configs = (
        {"solver": "rk-4", "fixed_steps": 10},
        {"solver": "tsit5", "atol": 1e-3, "fixed_steps": 2},
        # {"solver": "dopri5", "atol": 1e-3, "fixed_steps": 2},
        {"solver": "rk-4", "fixed_steps": 2},
        # {"solver": "rk-4", "fixed_steps": 5},
    )

    # train_all_pan(pan_configs, epochs=20, sensitivity="adjoint", test=True)
    train_all_shooting(shoot_configs, epochs=20, sensitivity="adjoint", test=True)
    train_all_pan(pan_configs, epochs=20, sensitivity="autograd", test=True)
    train_all_shooting(shoot_configs, epochs=20, sensitivity="adjoint", test=True)
