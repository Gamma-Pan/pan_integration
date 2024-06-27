import torch
from torch import nn
from torch.nn import functional as F
from torchdyn.core import NeuralODE
from torchdyn.datasets import ToyDataset
import matplotlib.pyplot as plt
import lightning as lit
from pan_integration.core.pan_ode import PanODE
from pan_integration.utils.plotting import wait, VfPlotter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = ToyDataset()
X, y = d.generate(n_samples=10_000, dataset_type="moons", noise=0.2)

torch.manual_seed(14)

from torch.utils.data import Dataset

X = torch.Tensor(X).to(device)
y = torch.Tensor(y).to(device)
train_dataset = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

train_acc = []

class LitLearner(lit.LightningModule):
    def __init__(self, model, t_span):
        super().__init__()
        self.model = model
        self.t_span = t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        _, traj = self.model(
            X,
            self.t_span,
        )
        y_hat = traj[-1]
        acc = torch.sum(y == torch.max(y_hat, dim=1)[1]).item() / y.shape[0]
        loss = nn.CrossEntropyLoss()(y_hat, y)
        train_acc.append(acc)
        self.log("loss", loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        self.log("nfe", self.model.vf.nfe, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_dataloader(self):
        return train_loader


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, 2)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        return self.lin2(F.tanh(self.lin1(x)))


vf = VF().to(device)

model_pan = PanODE(
    vf,
    solver={"num_coeff_per_dim": 16, "patience": 15, "tol": 1e-3},
    sensitivity="adjoint",
    device=device,
)

model_tsit = NeuralODE(
    vf,
    solver="tsit5",
    sensitivity="adjoint",
    atol=1e-3,
    rtol=1e-3,
)

model = model_tsit
# model = model_pan
t_span = torch.linspace(0, 1, 2).to(device)

lit_learner = LitLearner(model, t_span)
trainer = lit.Trainer(max_epochs=3, logger=False)
trainer.fit(lit_learner)

with torch.no_grad():
    X = X.cpu()
    y = y.cpu()
    fig, ax = plt.subplots()
    indices = torch.randperm(X.shape[0])[:100]
    ax.scatter(*X[indices].unbind(-1), c=y[indices], alpha=0.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plotter = VfPlotter(vf, existing_axes=ax)
    plotter.solve_ivp(
        torch.linspace(0, 1, 100),
        X[indices][y[indices].to(torch.bool)],
        set_lims=True,
        plot_kwargs=dict(color="orange", alpha=0.1),
        end_point = True
    )

    plotter.solve_ivp(
        torch.linspace(0, 1, 100),
        X[indices][torch.logical_not(y[indices].to(torch.bool))],
        set_lims=True,
        plot_kwargs=dict(color="purple", alpha=0.1),
        end_point=True,

    )
    plt.show()
