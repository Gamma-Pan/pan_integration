import glob
import torch
import torchdyn.utils
from torch import nn, tensor
import torch.utils.data as data
from torch.nn.functional import sigmoid

from torchdyn.datasets import ToyDataset
from torchdyn.models import NeuralODE

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("TkAgg")


class Learner(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        t_span: torch.Tensor = tensor([0.0, 1.0]),
    ):
        super().__init__()
        self.t_span, self.model = t_span, model
        self.linear = nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x, self.t_span)
        logits = sigmoid(self.linear(y_hat[-1]))[:, 0]
        nfe = self.model.vf.nfe
        loss = nn.BCELoss()(logits, y)

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x, self.t_span)
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
    train_dataset = data.TensorDataset(X_train, y_train)
    train_loader = data.DataLoader(train_dataset, batch_size=len(X), shuffle=True)

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

    model = NeuralODE(
        f,
        sensitivity="adjoint",
        solver="tsit5",
        atol=1e-3,
        rtol=1e-3,
        return_t_eval=False,
    ).to(device)

    learner = Learner(model)
    trainer = L.Trainer(max_epochs=200)  # callbacks=TQDMProgressBar(refresh_rate=5))
    trainer.fit(learner, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ckpt_path = glob.glob("lightning_logs/version_0/checkpoints/*.ckpt")
    # model = NeuralODE.load_from_checkpoint(ckpt_path[0])

    model.eval()
    t_span = torch.linspace(0.0, 1.0, 100)
    trajectory = model(X_val.cpu(), t_span).detach()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    colors = ["blue", "orange"]

    for i in range(len(X_val)):
        ax.plot(
            t_span,
            trajectory[:, i, 0],
            trajectory[:, i, 1],
            color=colors[int(y_val[i].item())],
        )

    plt.show()
