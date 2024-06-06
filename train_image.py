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
from pan_integration.utils.lightning import PlotTrajectories, NfeMetrics, ProfilerCallback, LitOdeClassifier

import wandb

from copy import copy
import argparse
import glob
from typing import Union

from datetime import datetime
import multiprocessing as mp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_WORKERS =  p.cpu_count()
MAX_STEPS = -1
DATASET = "CIFAR10"


class Augmenter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        #self.conv1 = nn.Conv2d(3, channels - 3, 3, 1, 1)

    def forward(self, x):
        #x = torch.cat([x, self.conv1(x)], dim=1)
        return x


class VF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.nfe = 0

        self.norm1 = nn.GroupNorm(3,3)
        self.conv1 = nn.Conv2d(3, channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(62,62)
        self.conv3 = nn.Conv2d(62, 3,1)


    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = self.norm1(x)
        x = self.conv1(x)
        x = F.softplus(x)
        x = self.conv2(x)
        x = F.softplus(x)
        x = self.norm2(x)
        x = self.conv3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.lin1= nn.Linear(3*16,10)

    def forward(self, x):
        # x = self.drop(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin1(x)
        return x


# dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


def run(
    name,
    mode=Union["pan", "shoot"],
    solver_config=None,
    log=False,
    points=10,
    epochs=50,
    max_steps=MAX_STEPS,
    profile=False,
    test=False,
):
    vf = VF(channels=CHANNELS).to(device)
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

    embedding = Augmenter(channels=CHANNELS).to(device)
    classifier = Classifier(channels=CHANNELS).to(device)

    learner = LitOdeClassifier(t_span, embedding, ode_model, classifier)
    nfe_callback = NfeMetrics()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename=f'{name}_chkp_{datetime.now().strftime("%d_%m_%H_%M")}',
        monitor="val_acc",
        mode="min",
    )
    prof_callback = ProfilerCallback(fname=name)

    callbacks = [nfe_callback, checkpoint_callback]
    if profile:
        callbacks.append(prof_callback)

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
        traj_callback = PlotTrajectories(vf)
        callbacks.append(traj_callback)

    trainer = Trainer(
        max_epochs=epochs,
        enable_checkpointing=True,
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        max_steps=max_steps,
    )
    dmodule = CIFAR10DataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    trainer.fit(learner, datamodule=dmodule)
    if test:
        trainer.test(learner, datamodule=dmodule)

    if log and profile:
        profile_artf = wandb.Artifact(f"trace_{name}", type="profile")
        profile_artf.add_file(local_path=f"{name}.json")
        logger.experiment.log_artifact(profile_artf)
    elif log and not profile:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--log", default=False, type=bool)
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--channels", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--profile", default=False, type=bool)
    args = vars(parser.parse_args())
    WANDB_LOG = args["log"]
    TEST = args["test"]
    CHANNELS = args["channels"]
    BATCH_SIZE = args["batch_size"]
    EPOCHS = args["epochs"]
    MAX_STEPS = args["max_steps"]
    PROFILE = args["profile"]

    configs = (
        dict(
            name="dopri",
            mode="shoot",
            solver_config={"solver": "dopri5", "atol": 1e-4, "rtol": 1e-4},
            log=WANDB_LOG,
            epochs=EPOCHS,
            profile=PROFILE,
            test=TEST,
            max_steps=MAX_STEPS,
        ),
        dict(
            name="pan_32_32",
            mode="pan",
            solver_config={
                "num_coeff_per_dim": 32,
                "num_points": 32,
                "deltas": (1e-4, -1),
                "max_iters": (30, 0),
            },
            log=WANDB_LOG,
            epochs=EPOCHS,
            profile=PROFILE,
            test=TEST,
            max_steps=MAX_STEPS
        ),
        # dict(
        #     name="rk4-10",
        #     mode="shoot",
        #     solver_config={"solver": "rk-4"},
        #     log=WANDB_LOG,
        #     epochs=EPOCHS,
        #     test=TEST,
        #     max_steps=MAX_STEPS
        # ),
    )

    print(BATCH_SIZE)
    for config in configs:
        run(**config)
