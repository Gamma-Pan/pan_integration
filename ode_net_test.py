import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchdyn.core import NeuralODE
from nde_squared.data import mnist_dataloaders

device = torch.device("cuda:0")


class Learner(pl.LightningModule):
    def __init__(self, t_span: torch.Tensor, ode_net: nn.Module):
        super().__init__()
        self.ode_net, self.t_span = ode_net, t_span
        self.fc = nn.Sequential(nn.Linear(28**2, 10))

    def forward(self, x):
        t_eval, traj = self.ode_net(x)
        y_hat = torch.flatten(traj[-1], start_dim=1)
        logits = self.fc(y_hat)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, traj = self.ode_net(x)
        y_hat = torch.flatten(traj[-1], start_dim=1)
        logits = self.fc(y_hat)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("loss", loss)
        return {"loss": loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


# first define f (the neural net) and then the neural ODE module
f = nn.Sequential(
    nn.Conv2d(1, 16, 5, 1, 2),
    nn.Tanh(),
    nn.BatchNorm2d(16),
    nn.Conv2d(16, 32, 3, 1, 1),
    nn.Tanh(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 1, 1, 1, 0),
)
ode_model = NeuralODE(f, sensitivity="adjoint", solver="dopri5")

t_span = torch.linspace(0, 1, 100)

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="Testing")
    pl_module = Learner(t_span, ode_model)
    train_loader, *rest = mnist_dataloaders(train=True)
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
    trainer.fit(pl_module, train_dataloaders=train_loader, )
