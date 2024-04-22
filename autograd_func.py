import torch
from torch import nn, tensor
from pan_integration.solvers.pan_integration import make_pan_adjoint
from torchdyn.core import NeuralODE
from torchdyn.datasets import ToyDataset
from lightning.pytorch import LightningModule, Trainer

torch.manual_seed(42)


class Learner(LightningModule):
    def __init__(
        self, model: nn.Module, t_span: torch.Tensor = torch.linspace(0, 1, 10)
    ):
        super().__init__()
        self.t_span, self.model = t_span, model
        self.linear = nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        logits = torch.nn.functional.sigmoid(self.linear(y_hat[-1]))
        loss = nn.BCELoss()(logits[..., 0], y)

        nfe = self.model.vf.nfe

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        logits = torch.nn.functional.sigmoid(self.linear(y_hat[-1]))
        loss = nn.BCELoss()(logits[..., 0], y)

        preds = logits[..., 0] > 0.5
        acc = torch.sum(preds == y) / y.shape[0]

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)


class PanODE(nn.Module):
    def __init__(self, vf, num_coeff_per_dim, num_points):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.pan_int = make_pan_adjoint(
            self.vf, self.thetas, num_coeff_per_dim, num_points
        )

    def forward(self, y_init, t):
        t_eval, traj = self.pan_int(y_init, t)
        return t_eval, traj


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 512),
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        self.nfe = 0

    def forward(self, t, y, *args, **kwargs):
        self.nfe += 1
        return self.seq(y)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_init = torch.rand(10, 2).requires_grad_(True).to(device)

    vf = VF().to(device)
    t_span = torch.linspace(0, 1, 5).to(device)
    model = PanODE(vf, num_coeff_per_dim=20, num_points=20)
    # model = NeuralODE(
    #     vf, sensitivity="adjoint", solver="tsit5", atol=1e-3, rtol=1e-3
    # ).to(device)

    # _, yS = model(y_init, t_span)
    # yT = yS[-1]
    # L = torch.sum(yT)
    # L.backward()

    d = ToyDataset()
    X_t, yn_t = d.generate(n_samples=10_000, dataset_type="moons", noise=0.4)
    X_train = torch.Tensor(X_t).to(device)
    y_train = torch.Tensor(yn_t.to(torch.float)).to(device)
    train = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    X_v, yn_v = d.generate(n_samples=100, dataset_type="moons", noise=0.4)
    X_val = torch.Tensor(X_v).to(device)
    y_val = torch.Tensor(yn_v.to(torch.float)).to(device)
    val = torch.utils.data.TensorDataset(X_val, y_val)
    valloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    learner = Learner(model, t_span).to(device)
    trainer = Trainer(max_epochs=3)
    trainer.fit(learner, train_dataloaders=trainloader, val_dataloaders=valloader)
