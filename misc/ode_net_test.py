import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import dataset, dataloader
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import Accuracy
from torchdyn.core import NeuralODE
from torchdyn.datasets import ToyDataset
from pan_integration.data import mnist_dataloaders
from typing import List, Any

mpl.use("TkAgg")

device = torch.device("cuda:0")


# first define f (the neural net) and then the neural ODE module
class VectorField(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dims=None, output_dim: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 16, 8]

        self.activation_fn = nn.ReLU()

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            self.activation_fn,
        )
        self.hidden = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(i, j), nn.BatchNorm1d(j), self.activation_fn)
                for i, j in zip(hidden_dims, hidden_dims[1:])
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim),
            # self.activation_fn
        )

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = nn.functional.dropout(x, p=0.05)
        x = self.output(x)
        return x


class LitLearner(pl.LightningModule):
    def __init__(
        self,
        t_span: torch.Tensor,
        ode_net: nn.Module,
        out_classes: int = 2,
        vf_dim: int = 2,
    ):
        super().__init__()
        self.ode_net, self.t_span = ode_net, t_span
        self.fc = nn.Sequential(nn.Linear(vf_dim, out_classes), nn.Sigmoid())
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=out_classes)

    def forward(self, x):
        t_eval, traj = self.ode_net(x)
        y_hat = torch.flatten(traj[-1], start_dim=1)
        logits = self.fc(y_hat)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        t_eval, traj = self.ode_net(x)
        y_hat = traj[-1]  # prediction is the last point of trajectory
        logits = self.fc(y_hat)
        labels = F.one_hot(y).double()

        loss = self.loss_fn(logits, labels)

        # compare labels to index of max prediction
        accuracy = self.accuracy_metric(y, torch.max(logits, dim=1)[1])

        self.log_dict({"loss": loss, "accuracy": accuracy})
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


class VizVectorField(pl.Callback):
    def __init__(self, plot_interval=20):
        super().__init__()
        self.plot_interval = plot_interval

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # produce 100 random points and save them
        # TODO: use a variable number of classes and generator
        d = ToyDataset()
        num_samples = 100
        pl_module.X, pl_module.y = d.generate(
            n_samples=num_samples, noise=1e-1, dataset_type="moons"
        )
        # init figure and axis
        pl_module.fig_vf, pl_module.ax_vf = plt.subplots()

        min_x, min_y = torch.min(pl_module.X, dim=0)[0]
        max_x, max_y = torch.max(pl_module.X, dim=0)[0]

        min_x = min_x - (torch.sign(min_x) * min_x / 2)
        min_y = min_y - (torch.sign(min_y) * min_y / 2)

        max_x = max_x + (torch.sign(max_x) * max_x / 2)
        max_y = max_y + (torch.sign(max_y) * max_y / 2)

        pl_module.min_x, pl_module.max_x, pl_module.min_y, pl_module.max_y = (
            min_x,
            max_x,
            min_y,
            max_y,
        )

        pl_module.ax_vf.set_xlim([min_x, max_x])
        pl_module.ax_vf.set_ylim([min_y, max_y])

        pl_module.ax_vf.scatter(
            pl_module.X[:, 0],
            pl_module.X[:, 1],
            s=15,
            c=0.7 * pl_module.y,
            zorder=10,
            cmap=plt.colormaps["seismic"],
        )
        pl_module.lines = pl_module.ax_vf.plot([0], [0])
        pl_module.ends = pl_module.ax_vf.scatter(
            torch.rand(num_samples),
            torch.rand(num_samples),
            s=15,
            c=0.7 * pl_module.y,
            zorder=10,
            cmap=plt.colormaps["bwr"],
        )
        pl_module.quiver = pl_module.ax_vf.quiver(
            torch.zeros(num_samples, num_samples), torch.zeros(num_samples, num_samples)
        )

        plt.pause(0.5)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        with torch.no_grad():
            for line in pl_module.lines:
                # erase previous lines
                line.remove()

            pl_module.quiver.remove()

            sample_X, sample_y = pl_module.X, pl_module.y
            t_span = torch.linspace(0, 1, 100).cuda()

            # set to eval mode
            pl_module.ode_net.eval()
            t_eval, trajectories = pl_module.ode_net(sample_X.cuda(), t_span)

            trajectories = trajectories.detach().cpu()

            # newmin_x, newmin_y = torch.min(torch.min(trajectories, dim=0)[0], dim=0)[0]
            # newmax_x, newmax_y = torch.max(torch.max(trajectories, dim=0)[0], dim=0)[0]
            #
            # if newmin_x < 0.9 * pl_module.min_x:
            #     pl_module.min_x = newmin_x * 0.9
            # if newmin_y < 0.9 * pl_module.min_y:
            #     pl_module.min_y = newmin_y * 0.9
            # if newmax_x > 1.1 * pl_module.max_x:
            #     pl_module.max_x = newmax_x * 1.1
            # if newmax_y > 1.1 * pl_module.max_y:
            #     pl_module.max_y = newmax_y * 1.1
            # pl_module.ax_vf.set_xlim([pl_module.min_x, pl_module.max_x])
            # pl_module.ax_vf.set_ylim([pl_module.min_y, pl_module.max_y])

            vf_steps = 30
            # grid for vector field
            X_grid, Y_grid = torch.meshgrid(
                [
                    torch.linspace(pl_module.min_x, pl_module.max_x, vf_steps),
                    torch.linspace(pl_module.min_y, pl_module.max_y, vf_steps),
                ],
                indexing="xy",
            )

            vf_batch = torch.hstack((X_grid.reshape(-1, 1), Y_grid.reshape(-1, 1)))
            f = pl_module.ode_net.vf(0, vf_batch.cuda()).cpu()

            X_arrow = f[:, 0].reshape((vf_steps, vf_steps))
            Y_arrow = f[:, 1].reshape((vf_steps, vf_steps))
            pl_module.quiver = pl_module.ax_vf.quiver(
                X_grid, Y_grid, X_arrow, Y_arrow, color="#555555aa", zorder=5
            )

            # set model to train mode (is this necessary?)
            pl_module.ode_net.train()

            pl_module.lines = []
            pl_module.lines += pl_module.ax_vf.plot(
                trajectories[:, :50, 0],
                trajectories[:, :50, 1],
                color="#34397855",
                linewidth=2,
            )
            pl_module.lines += pl_module.ax_vf.plot(
                trajectories[:, 50:, 0],
                trajectories[:, 50:, 1],
                color="#9f783455",
                linewidth=2,
            )
            pl_module.ends.set_offsets(trajectories[-1, :, :])

            max_steps = len(trainer.train_dataloader)
            pl_module.fig_vf.suptitle(
                f"Epoch: {trainer.current_epoch} "
                f"   step: {batch_idx}/{max_steps} "
                f'   accuracy: {trainer.callback_metrics["accuracy"]:2.2f}'
            )

            plt.pause(0.2)


def train():
    wandb_logger = None
    # wandb_logger = WandbLogger(project="Testing")

    t_span = torch.linspace(0, 1, 100)
    f = VectorField()
    ode_model = NeuralODE(f, sensitivity="adjoint", solver="rk4")
    pl_module = LitLearner(t_span, ode_model)
    my_callback = VizVectorField()

    trainer = pl.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        callbacks=[my_callback],
        log_every_n_steps=10,
    )

    d = ToyDataset()
    num_samples = 10000
    X, y = d.generate(n_samples=num_samples, noise=2e-1, dataset_type="moons")
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    trainer.fit(pl_module, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
