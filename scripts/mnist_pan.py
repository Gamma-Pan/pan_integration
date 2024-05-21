import torch
from torch import nn, tensor

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
NUM_WORKERS = 12


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0
        # self.norm1 = nn.GroupNorm(4, 64)
        # self.norm2 = nn.GroupNorm(4, 64)
        # self.norm3 = nn.GroupNorm(4, 64)

        # self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # self.norm1 = nn.BatchNorm1d(BATCH_SIZE)
        # self.norm2 = nn.BatchNorm1d(BATCH_SIZE)
        # self.norm3 = nn.BatchNorm1d(BATCH_SIZE)
        self.lin1 = nn.Linear(784, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, 256)
        self.lin5 = nn.Linear(256, 784)

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        # x = self.conv1(F.relu(self.norm1(x)))
        # x = self.conv2(F.relu(self.norm2(x)))
        # x = self.norm3(x)

        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = nn.functional.relu(self.lin3(x))
        x = nn.functional.relu(self.lin4(x))
        x = nn.functional.relu(self.lin5(x))

        return x


embedding = nn.Sequential(
    # nn.Conv2d(1, 64, 3, 1),
    # nn.GroupNorm(8, 64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 4, 2, 1),
    # nn.GroupNorm(8, 64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 4, 2, 1),
    # nn.Linear(28*28, 512),
    # nn.Tanh(),
    nn.Flatten(start_dim=1),
).to(device)

classifier = nn.Sequential(
    # nn.GroupNorm(8, 64),
    # nn.ReLU(),
    # nn.AdaptiveAvgPool2d((1, 1)),
    # nn.Flatten(),
    nn.Linear(784, 10),
)

vf = VF().to(device)


def train_mnist_ode(
    ode_model,
    epochs=10,
    test=False,
    wandb=False
):
    if wandb:
        logger = WandbLogger(project="pan_integration", name="test_run", log_model=False)
        logger.experiment.config["batch_size"] = BATCH_SIZE

    learner = LitOdeClassifier(embedding, ode_model, classifier, t_span=t_span)
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

    if wandb:
        wandb.finish()

if __name__ == "__main__":
    t_span = torch.linspace(0, 1, 2).to(device)

    solver = PanZero(
        32, 64, delta=1e-3, max_iters=30, t_lims=(t_span[0], t_span[-1]), device=device
    )
    solver_adjoint = PanZero(
        32, 64, delta=1e-3, max_iters=30, t_lims=(t_span[-1], t_span[0]), device=device
    )
    ode_model = PanODE(vf, solver, solver_adjoint).to(device)

    # tsit_args = dict(solver="tsit5", sensitivity="autograd")
    # ode_model = NeuralODE(vf, **tsit_args)

    # solver_args = dict(solver="mszero", maxiter=4, fine_steps=4)
    # ode_model = MultipleShootingLayer(vf, **solver_args)

    train_mnist_ode(ode_model, test=True, wandb=True)

