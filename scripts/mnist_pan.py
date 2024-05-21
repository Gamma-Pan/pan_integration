import torch
from torch import nn, tensor
from lightning.pytorch import LightningModule, Trainer
from pan_integration.data import MNISTDataModule
from torchdyn.core import MultipleShootingLayer, NeuralODE
from pan_integration.numerics.pan_solvers import PanZero, make_pan_adjoint

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 12


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
        self.B_prev = None
        # self.save_hyperparameters()

    def _common_step(self, batch, batch_idx, log=True, update_B=False):
        x, y = batch
        x_em = self.embedding(x)
        if isinstance(self.ode_model, PanODE):
            t_eval, y_hat, metrics, B = self.ode_model(x_em, self.t_span, B_init=self.B_prev)
            if update_B:
                with torch.no_grad():
                    self.B_prev = B[...,2:]
            if log:
                self.log("zero_B_delta", float(metrics[0]), prog_bar=True)
                self.log("zero_iters", float(metrics[1]), prog_bar=True)
        else:
            t_eval, y_hat = self.ode_model(x_em, self.t_span)

        logits = self.classifier(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        _, preds = torch.max(logits, dim=1)
        acc = torch.sum(preds == y) / y.shape[0]
        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx, update_B=True)
        nfe = self.ode_model.vf.nfe

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx, log=False)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.ode_model.parameters(), lr=0.0005)


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        solver,
        solver_adjoint,
    ):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.solver = solver
        self.solver_adjoint = solver_adjoint

        self.pan_int = make_pan_adjoint(
            self.vf,
            self.thetas,
            self.solver,
            self.solver_adjoint,
        )

    def forward(self, y_init, t_eval, B_init=None, *args, **kwargs):
        traj, B, metrics = self.solver.solve(self.vf, t_eval, y_init, B_init=B_init)
        return t_eval, traj, metrics, B


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
    learner,
    trainer,
    dmodule,
    test=False,
):
    trainer.fit(learner, datamodule=dmodule)
    if test:
        trainer.test(learner, datamodule=dmodule)


if __name__ == "__main__":
    t_span = torch.linspace(0, 1, 10).to(device)

    tsit_args = dict(solver="tsit5", sensitivity="autograd")

    solver = PanZero(
        32, 32, delta=1e-3, max_iters=30, t_lims=(t_span[0], t_span[-1]), device=device
    )
    solver_adjoint = PanZero(
        32, 32, delta=1e-3, max_iters=30, t_lims=(t_span[-1], t_span[0]), device=device
    )

    ode_model = NeuralODE(vf, **tsit_args)
    # ode_model = PanODE(vf, solver, solver_adjoint).to(device)

    # solver_args = dict(solver="mszero", maxiter=4, fine_steps=4)
    # ode_model = MultipleShootingLayer(vf, **solver_args)

    learner = Learner(embedding, ode_model, classifier, t_span=t_span)
    dmodule = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    trainer = Trainer(
        max_epochs=2, enable_checkpointing=False, fast_dev_run=False, accelerator="gpu"
    )

    train_mnist_ode(learner, trainer, dmodule, test=True)
