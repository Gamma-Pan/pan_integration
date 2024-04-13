import glob
import torch
import torchdyn.utils
from torch import nn, tensor, linspace
import torch.utils.data as data
from torch.nn.functional import sigmoid

from torchdyn.datasets import ToyDataset
from torchdyn.models import NeuralODE, MultipleShootingLayer
from torchdyn.numerics.solvers.ode import MultipleShootingDiffeqSolver

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, EarlyStopping

import matplotlib.pyplot as plt
import matplotlib as mpl

from pan_integration.solvers.pan_integration import _cheb_phis
from torch import stack, tensor, cat
from torch.linalg import inv

mpl.use("TkAgg")

from pan_integration.utils import plotting


class LSZero(MultipleShootingDiffeqSolver):
    def __init__(self, num_coeff_per_dim, num_points, etol=1e-5, callback=None):
        super().__init__("euler", "euler")
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_points
        self.etol = etol
        self.callback = None

    def sync_device_dtype(self, x, t_span):
        return x, t_span

    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        y_init = x
        device = y_init.device
        f_init = f(t_span, y_init)
        t_lims = [t_span[0], t_span[-1]]

        batches, dims = y_init.shape
        C = torch.rand(batches, self.num_coeff_per_dim - 2, dims, device=device)

        t = -torch.cos(torch.pi * (torch.arange(self.num_points) / self.num_points) ).to(device)
        d = t_lims[1] * torch.diff(torch.cat((t, tensor([1.0], device=device))))[:, None]

        Phi, DPhi = _cheb_phis(
            self.num_points, self.num_coeff_per_dim, t_lims, device=device
        )

        inv0 = inv(stack((Phi[:, 0, [0, 1]], DPhi[:, 0, [0, 1]]), dim=1))
        Phi_aT = DPhi[:, :, [0, 1]] @ inv0 @ stack((y_init, f_init), dim=1)
        Phi_bT = (
            -DPhi[:, :, [0, 1]] @ inv0 @ cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1)
            + DPhi[:, :, 2:]
        )
        l = lambda C: inv0 @ (
            stack((y_init, f_init), dim=1)
            - cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1) @ C
        )

        Q = inv(Phi_bT.mT @ (d * Phi_bT))
        # MAIN LOOP
        for i in range(50):
            if self.callback is not None:
                self.callback(C.cpu())

            C_prev = C
            # C update
            C = Q @ (
                Phi_bT.mT @ (d * f(t_span, Phi @ cat((l(C), C), dim=1)))
                - Phi_bT.mT @ (d * Phi_aT)
            )

            if torch.norm(C - C_prev) < self.etol:
                break

        # refine with newton

        Phi_out, _ = _cheb_phis(
            len(t_span), self.num_coeff_per_dim, t_lims, include_end=True, device=device
        )

        approx = Phi_out @ cat((l(C), C), dim=1)
        # return (time,batch,dim)
        return approx.transpose(0, 1)


class Learner(L.LightningModule):
    def __init__(
        self, model: nn.Module, t_span: torch.Tensor = torch.linspace(0, 1, 10)
    ):
        super().__init__()
        self.t_span, self.model = t_span, model
        self.linear = nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        logits = sigmoid(self.linear(y_hat[-1]))[:, 0]

        nfe = self.model.vf.nfe
        loss = nn.BCELoss()(logits, y)

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        logits = sigmoid(self.linear(y_hat[-1]))[:, 0]

        labels = logits > 0.5
        acc = torch.sum(labels == y) / 100
        loss = nn.BCELoss()(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    d = ToyDataset()
    X, y = d.generate(n_samples=10_000, dataset_type="moons", noise=0.1)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.Tensor(y.to(torch.float32)).to(device)

    colors = ["blue", "orange"]

    train_dataset = data.TensorDataset(X_train, y_train)
    train_loader = data.DataLoader(train_dataset, batch_size=1000, shuffle=True)

    X, y = d.generate(n_samples=100, dataset_type="moons", noise=0.1)

    X_val = torch.Tensor(X).to(device)
    y_val = torch.Tensor(y.to(torch.float32)).to(device)
    val_dataset = data.TensorDataset(X_val, y_val)
    val_loader = data.DataLoader(val_dataset, batch_size=len(X))

    f = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2),
    )
    solver = LSZero(30, 50, etol=1e-3)

    model = MultipleShootingLayer(
        f,
        solver=solver,
        # solver='zero'
        # return_t_eval=False
    ).to(device)

    learner = Learner(model)
    trainer = L.Trainer(
        max_epochs=200,
        callbacks=[
            EarlyStopping(
                monitor="val_acc", stopping_threshold=0.99, mode="max", patience=100
            )
        ],
    )
    trainer.fit(learner, train_dataloaders=train_loader, val_dataloaders=val_loader)
