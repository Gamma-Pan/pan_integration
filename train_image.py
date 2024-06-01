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

from pan_integration.data import MNISTDataModule, CIFAR10DataModule
from pan_integration.core.pan_ode import PanODE, PanSolver
from pan_integration.utils.lightning import (
    LitOdeClassifier,
    NfeMetrics,
    ProfilerCallback,
)

import wandb

from copy import copy
import argparse
import glob
from typing import Union

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

import multiprocessing as mp

NUM_WORKERS = mp.cpu_count()
CHANNELS = 5
NUM_GROUPS = 1
WANDB_LOG = False
EPOCHS = 100
MAX_STEPS = -1
TEST = True
DATASET = 'CIFAR10'


class Augmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, CHANNELS, 3, 1, 1)
        self.norm1 = nn.GroupNorm(NUM_GROUPS, CHANNELS)

    def forward(self, x):
        x = F.tanh(self.norm1(self.conv1(x)))
        return x


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, 1)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, 1)
        self.conv3 = nn.Conv2d(CHANNELS, CHANNELS, 3, 1, 1)
        self.norm1 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.norm2 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.norm3 = nn.GroupNorm(NUM_GROUPS, CHANNELS)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.tanh(self.norm3(self.conv3(x)))
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 1, 3, 1, 1)
        self.linear1 = nn.Linear(32 * 32, 10)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x


# dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
dmodule = CIFAR10DataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


def run(
    name,
    mode=Union["pan", "shoot"],
    solver_config=None,
    log=False,
    points=10,
    epochs=50,
    max_steps=MAX_STEPS,
    profile=False,
    test=TEST,
):
    vf = VF().to(device)
    t_span = torch.linspace(0, 1, 10, device=device)
    if mode == "pan":
        sensitivity = "adjoint"
        ode_model = PanODE(
            vf,
            t_span,
            solver=solver_config,
            solver_adjoint=solver_config,
            sensitivity=sensitivity,
        ).to(device)

    if mode == "shoot":
        sensitivity = "adjoint"
        ode_model = NeuralODE(vf, **solver_config, sensitivity=sensitivity).to(device)

    embedding = Augmenter().to(device)
    classifier = Classifier().to(device)

    logger = ()
    if log:
        logger = WandbLogger(project="pan_integration", name=name, log_model=False)
        logger.experiment.config.update(solver_config)
        logger.experiment.config.update(
            {
                "type": mode,
                "sensitivity": sensitivity,
                "points": points,
                "dataset": DATASET,
            }
        )

    learner = LitOdeClassifier(t_span, embedding, ode_model, classifier)
    nfe_callback = NfeMetrics()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        save_top_k=1,
        monitor="val_acc_epoch",
        mode="min",
    )
    prof_callback = ProfilerCallback()

    callbacks = [nfe_callback]
    if profile:
        callbacks.append(prof_callback)

    trainer = Trainer(
        max_epochs=epochs,
        enable_checkpointing=True,
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        max_steps=max_steps,
    )

    trainer.fit(learner, datamodule=dmodule)
    if test:
        trainer.test(learner, datamodule=dmodule)

    if log and profile:
        profile_artf = wandb.Artifact(f"trace_{name}", type="profile")
        profile_artf.add_file(local_path="./trace.json")
        logger.experiment.log_artifact(profile_artf)
    elif log and not profile:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--log", default=False, type=bool)
    args = vars(parser.parse_args())
    WANDB_LOG = args["log"]

    configs = (
        dict(
            name="pan_16_16",
            mode="pan",
            solver_config={
                "num_coeff_per_dim": 20,
                "num_points": 20,
                "deltas": (1e-3, -1),
                "max_iters": (30, 0),
            },
            log=WANDB_LOG,
            epochs=EPOCHS,
        ),
        dict(
            name="rk4-10",
            mode="shoot",
            solver_config={"solver": "rk-4"},
            log=WANDB_LOG,
            epochs=EPOCHS,
        ),
        dict(
            name="tsit5",
            mode="shoot",
            solver_config={"solver": "tsit5", "atol": 1e-3},
            log=WANDB_LOG,
            epochs=EPOCHS,
        ),
    )

    for config in configs:
        run(**config)
