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
BATCH_SIZE = 128

import multiprocessing as mp

NUM_WORKERS = mp.cpu_count()
CHANNELS = 1
NUM_GROUPS = 1
WANDB_LOG = False
EPOCHS = 60


class Augmenter(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, CHANNELS, 3, 1, 1)
        # self.norm1 = nn.GroupNorm(NUM_GROUPS, CHANNELS)

    def forward(self, x):
        dims = x.shape
        x = torch.cat(
            [
                x.view(dims[0], -1),
                torch.zeros((dims[0], 1000 - 28**2), device=device),
            ],
            dim=-1,
        )
        return x


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1000, 1000)
        self.lin2 = nn.Linear(1000, 1000)
        self.norm1 = nn.LayerNorm(1000)
        self.norm2 = nn.LayerNorm(1000)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = F.relu(self.norm1(self.lin1(x)))
        x = F.tanh(self.norm2(self.lin2(x)))
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1000, 10)
        self.drop = nn.Dropout(0.01)

    def forward(self, x):
        x = self.lin1(x)
        x = self.drop(x)
        return x


dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


def run(
    name,
    mode=Union["pan", "shoot"],
    solver_config=None,
    log=False,
    points=10,
    epochs=50,
    max_steps=-1,
    profile=False,
    test=True,
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
        sensitivity = "interpolated_adjoint"
        ode_model = NeuralODE(vf, **solver_config, sensitivity=sensitivity).to(device)

    embedding = Augmenter(CHANNELS)
    classifier = Classifier()

    logger = ()
    if log:
        logger = WandbLogger(project="pan_integration", name=name, log_model=False)
        logger.experiment.config.update(solver_config)
        logger.experiment.config.update(
            {"type": mode, "sensitivity": sensitivity, "points": points}
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
                "num_coeff_per_dim": 16,
                "num_points": 16,
                "deltas": (1e-3, -1),
                "max_iters": (20, 0),
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
            solver_config={
                "solver": "tsit5",
            },
            log=WANDB_LOG,
            epochs=EPOCHS,
        ),
    )

    for config in configs:
        run(**config)
