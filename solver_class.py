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
        x.reshape(-1, 28 * 28)
        t_eval, y_hat = self.model(x, self.t_span)
        logits = sigmoid(self.linear(y_hat[-1]))[:, 0]

        nfe = self.model.vf.nfe
        loss = nn.BCELoss()(logits, y)

        self.nfes.append(nfe)
        self.losses.append(loss.detach().cpu())

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x.reshape(-1,28*28)
        t_eval, y_hat = self.model(x, self.t_span)
        logits = sigmoid(self.linear(y_hat[-1]))[:, 0]

        labels = logits > 0.5
        acc = torch.sum(labels == y) / 100
        loss = nn.BCELoss()(logits, y)

        self.vals.append(acc.detach().cpu())
        self.val_nfes.append(self.model.vf.nfe)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class F(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(28 * 28, 256)
            self.lin2 = nn.Linear(256, 256)
            self.lin3 = nn.Linear(256, 28 * 28)

        def forward(self, t, x):
            x = tanh(self.lin1(x))
            x = tanh(self.lin2(x))
            x = tanh(self.lin3(x))

    f = F()

    solver = LSZero(30, 50, etol=1e-6)

    model = MultipleShootingLayer(
        f,
        solver="zero"
        # return_t_eval=False
    ).to(device)

    learner = Learner(model)
    trainer = L.Trainer(
        max_epochs=3,
        # callbacks=[
        #     EarlyStopping(
        #         monitor="val_acc", stopping_threshold=0.99, mode="max", patience=100
        #     )
        # ],
    )
    dmodule = MNISTDataModule()
    trainer.fit(learner, datamodule=dmodule)
