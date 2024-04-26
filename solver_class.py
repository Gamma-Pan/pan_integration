import glob
import torch
import torchdyn.utils
from torch import nn, tensor, linspace
import torch.utils.data as data
from torch.nn.functional import sigmoid

from torchdyn.datasets import ToyDataset
from torchdyn.models import NeuralODE, MultipleShootingLayer
from torchdyn.numerics.solvers.ode import MultipleShootingDiffeqSolver

from pan_integration.data import MNISTDataModule

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, EarlyStopping

import matplotlib.pyplot as plt
import matplotlib as mpl

from torch import stack, tensor, cat
from torch.linalg import inv

from torch.nn.functional import tanh

mpl.use("TkAgg")

from pan_integration.utils import plotting
from pan_integration.solvers.pan_integration import pan_int


class LSZero(MultipleShootingDiffeqSolver):
    def __init__(self, num_coeff_per_dim, num_points, etol=1e-3):
        super().__init__(coarse_method="euler", fine_method="euler")
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_points
        self.etol = etol

    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        traj = pan_int(
            f,
            t_span,
            x,
            num_points=self.num_points,
            num_coeff_per_dim=self.num_coeff_per_dim,
            etol_ls=self.etol,
        )
        return traj


class Learner(L.LightningModule):
    def __init__(
        self, model: nn.Module, t_span: torch.Tensor = torch.linspace(0, 1, 10)
    ):
        super().__init__()
        self.t_span, self.model = t_span, model
        self.linear = nn.Linear(28 * 28, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x.reshape(-1, 28*28), self.t_span)
        logits = self.linear(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        nfe = self.model.vf.nfe

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x.reshape(-1, 28*28), self.t_span)
        logits = self.linear(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        _, preds= torch.max(logits, dim=1 )
        acc = torch.sum(preds== y) / y.shape[0]

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class F(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(28 * 28, 256)
            self.lin2 = nn.Linear(256, 256)
            self.lin3 = nn.Linear(256, 28 * 28)
            self.nfe = 0

        def forward(self, t, x):
            self.nfe +=1
            x = tanh(self.lin1(x))
            x = tanh(self.lin2(x))
            # x = nn.BatchNorm1d(64)(x)
            x = tanh(self.lin3(x))
            return x

    f = F()

    solver = LSZero(50, 100, etol=1e-6)

    model = MultipleShootingLayer(
        f,
        # solver="zero",
        solver=solver,
        # return_t_eval=False
    ).to(device)

    learner = Learner(model)
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[
            EarlyStopping(
                monitor="val_acc", stopping_threshold=0.9, mode="max", patience=5
            )
        ],
    )
    # dmodule = MNISTDataModule(batch_size=64, num_workers=12)
    # trainer.fit(learner, datamodule=dmodule)
