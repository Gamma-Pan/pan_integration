import torch
from torch import nn, tensor
from torch.nn import functional as F
from pan_integration.numerics.functional import make_pan_adjoint
from lightning.pytorch import LightningModule, Trainer
from pan_integration.data import MNISTDataModule
from torchdyn.core import MultipleShootingLayer, NeuralODE
from torchdyn.numerics.solvers.ode import MultipleShootingDiffeqSolver, MSZero

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


class Learner(LightningModule):
    def __init__(
        self,
        embedding: nn.Module,
        ode_model: nn.Module,
        classifier: nn.Module,
        t_span: torch.Tensor = torch.linspace(0, 1, 10),
    ):
        super().__init__()
        self.t_span, self.ode_model = t_span, ode_model
        self.embedding = embedding
        self.classifier = classifier
        # self.save_hyperparameters()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x_em = self.embedding(x)
        # t_eval, y_hat, metrics = self.ode_model(x_em, self.t_span)
        t_eval, y_hat= self.ode_model(x_em, self.t_span)
        # self.log("solver_loss", metrics[0], prog_bar=True)
        # self.log("zero_iters", metrics[1], prog_bar=True)

        logits = self.classifier(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        _, preds = torch.max(logits, dim=1)
        acc = torch.sum(preds == y) / y.shape[0]
        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)
        nfe = self.ode_model.vf.nfe

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.ode_model.parameters(), lr=0.0005)


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        num_coeff_per_dim,
        num_points,
        tol_zero=1e-3,
        tol_one=1e-5,
        max_iters_zero=10,
        max_iters_one=10,
        optimizer_class=None,
        optimizer_params=None,
        init="random",
        coarse_steps=5,
        callback=None,
        metrics=True,
    ):
        super().__init__()

        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.pan_int = make_pan_adjoint(
            self.vf,
            self.thetas,
            num_coeff_per_dim,
            num_points,
            tol_zero=tol_zero,
            tol_one=tol_one,
            max_iters_zero=max_iters_zero,
            max_iters_one=max_iters_one,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            init=init,
            coarse_steps=coarse_steps,
            callback = callback,
            metrics = metrics
        )

    def forward(self, y_init, t):
        t_eval, traj, metrics = self.pan_int(y_init, t)
        return t_eval, traj, metrics


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
        self.lin1 = nn.Linear(256, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 256)

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        # x = self.conv1(F.relu(self.norm1(x)))
        # x = self.conv2(F.relu(self.norm2(x)))
        # x = self.norm3(x)

        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = nn.functional.relu(self.lin3(x))

        return x


embedding = nn.Sequential(
    # nn.Conv2d(1, 64, 3, 1),
    # nn.GroupNorm(8, 64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 4, 2, 1),
    # nn.GroupNorm(8, 64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 4, 2, 1),
    nn.Flatten(start_dim=1),
    nn.Linear(28*28, 256),
    nn.Sigmoid(),
).to(device)

classifier = nn.Sequential(
    # nn.GroupNorm(8, 64),
    # nn.ReLU(),
    # nn.AdaptiveAvgPool2d((1, 1)),
    # nn.Flatten(),
    nn.Linear(256, 10),
)

vf = VF().to(device)

def train_mnist_ode(
    mode: str, solver_args: dict, learner_args: dict, trainer_args: dict, test=False
):

    if mode == "pan":
        ode_model = PanODE(vf, **solver_args)
    if mode == "ms":
        ode_model = MultipleShootingLayer(vf, **solver_args)
    if mode == "stepping":
        ode_model = NeuralODE(vf, **solver_args)

    ode_model = ode_model.to(device)
    learner = Learner(embedding, ode_model, classifier, **learner_args)

    dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=12)

    trainer = Trainer(**trainer_args)
    trainer.fit(learner, datamodule=dmodule)
    trainer.test(learner, datamodule=dmodule)


if __name__ == "__main__":
    # mode = "ms"
    # solver_args = dict(solver="mszero", maxiter=4, fine_steps=4)

    mode = "pan"

    num_points = 64

    # solver_args = dict(
    #     num_coeff_per_dim=32,
    #     num_points=num_points,
    #     tol_zero=1e-3,
    #     max_iters_zero=20,
    #     max_iters_one=0,
    #     # init='euler',
    #     # coarse_steps=5,
    #     metrics = True
    # )

    mode='stepping'
    solver_args = dict(solver="tsit5")

    t_span = torch.linspace(0, 1, 10).to(device)
    learner_args = dict(t_span=t_span)

    trainer_args = dict(
        max_epochs=3, enable_checkpointing=False, fast_dev_run=False, accelerator="gpu"
    )
    train_mnist_ode(mode, solver_args, learner_args, trainer_args, test=True)
