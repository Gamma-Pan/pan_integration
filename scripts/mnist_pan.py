import torch
from torch import nn, tensor
from torch.nn import functional as F

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from torchdyn.core import MultipleShootingLayer, NeuralODE

from pan_integration.data import MNISTDataModule
from pan_integration.core.ode import PanODE, PanZero
from pan_integration.utils.lightning import LitOdeClassifier

import wandb

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

import multiprocessing as mp

NUM_WORKERS = mp.cpu_count()


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0
        self.norm1 = nn.GroupNorm(4, 64)
        self.norm2 = nn.GroupNorm(4, 64)
        self.norm3 = nn.GroupNorm(4, 64)

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # self.lin1 = nn.Linear(784, 256)
        # self.lin2 = nn.Linear(256, 256)
        # self.lin3 = nn.Linear(256, 256)
        # self.lin4 = nn.Linear(256, 256)
        # self.lin5 = nn.Linear(256, 784)

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1

        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        x = self.norm3(x)

        # x = nn.functional.relu(self.lin1(x))
        # x = nn.functional.relu(self.lin2(x))
        # x = nn.functional.relu(self.lin3(x))
        # x = nn.functional.relu(self.lin4(x))
        # x = nn.functional.relu(self.lin5(x))

        return x


# networks' architectures takes from here for comparison :
# https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
embedding = nn.Sequential(
    nn.Conv2d(1, 64, 3, 1),
    nn.GroupNorm(8, 64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 4, 2, 1),
    nn.GroupNorm(8, 64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 4, 2, 1),
    # nn.Flatten(start_dim=1),
).to(device)

classifier = nn.Sequential(
    nn.GroupNorm(8, 64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10),
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


class PanChainODE(PanODE):
    def __init__(self, vf, t_span, solver, solver_adjoint):
        super().__init__(vf, t_span, solver, solver_adjoint)
        self.t_eval = t_span

    def forward(self, y_init, *args, **kwargs):
        _, traj, _, = super().forward(y_init, *args, **kwargs)
        return traj[-1]


class ChainODE(NeuralODE):
    def __init__(self, vf):
        super().__init__(vf, solver='rk-4', return_t_eval=False)

    def forward(self,x):
        return super().forward(x)[-1]


if __name__ == "__main__":
    t_span = torch.linspace(0, 1, 2).to(device)
    num_coeff_per_dim, num_points = 16, 16
    odes = []
    for _ in range(6):
        solver = PanZero(num_coeff_per_dim, num_points, delta=1e-3, max_iters=20, device=device, )
        solver_adjoint = PanZero(num_coeff_per_dim, num_points, delta=1e-3, max_iters=20, device=device, )
        ode_model = PanChainODE(VF().to(device), t_span, solver, solver_adjoint)
        # ode_model = ChainODE(VF().to(device))
        odes.append(ode_model)

    ode_seq = torch.nn.Sequential(*odes)

    logger = WandbLogger(project='pan_integration')
    logger.experiment.config["num_coeff_per_dim"] = num_coeff_per_dim
    logger.experiment.config["num_points"] = num_points
    train_mnist_ode(t_span, ode_seq, epochs=3, test=True, loggers=(logger,))
    wandb.finish()
