import torch
from torch import nn, tensor
from torch.nn.functional import tanh, relu
from pan_integration.solvers.pan_integration import make_pan_adjoint
from lightning.pytorch import LightningModule, Trainer
from pan_integration.data import MNISTDataModule
from torchdyn.core import MultipleShootingLayer, NeuralODE
from torchdyn.numerics.solvers.ode import MultipleShootingDiffeqSolver, MSZero

torch.manual_seed(42)


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
        t_eval, y_hat = self.ode_model(x_em, self.t_span)
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
        num_coeff_per_dim_adjoint=None,
        num_points_adjoint=None,
        etol_ls=1e-5,
    ):
        super().__init__()

        if num_coeff_per_dim_adjoint is None:
            num_coeff_per_dim_adjoint = num_coeff_per_dim
        if num_points_adjoint is None:
            num_points_adjoint = num_points

        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.pan_int = make_pan_adjoint(
            self.vf,
            self.thetas,
            num_coeff_per_dim,
            num_points,
            num_coeff_per_dim_adjoint,
            num_points_adjoint,
            etol_ls=etol_ls,
        )

    def forward(self, y_init, t):
        t_eval, traj = self.pan_int(y_init, t)
        return t_eval, traj


class VF(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.lin1 = nn.Linear(64, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = relu(self.lin1(x))
        x = relu(self.lin2(x))
        x = relu(self.lin3(x))
        # batchnorm wants (BxFxT)
        if len(x.shape) == 3:
            return self.batch_norm(x.permute(1,2,0)).permute(2,0,1)
        else:
            return self.batch_norm(x)


def train_mnist_ode(
    mode: str, solver_args: dict, learner_args: dict, trainer_args: dict, test=False
):
    vf = VF()
    embedding = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding_mode="zeros", padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=1, kernel_size=2, stride=2, padding=1),
        nn.Flatten(),  # (B,64)
    )
    classifier = nn.Sequential(nn.Linear(64, 10))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if mode == "pan":
        ode_model = PanODE(vf, **solver_args)
    if mode == "ms":
        ode_model = MultipleShootingLayer(vf, **solver_args)
    if mode == "stepping":
        ode_model = NeuralODE(**solver_args)

    learner = Learner(embedding, ode_model, classifier)

    dmodule = MNISTDataModule(batch_size=64)

    trainer = Trainer(**trainer_args)
    trainer.fit(learner, datamodule=dmodule)
    trainer.test(learner, datamodule=dmodule)


if __name__ == "__main__":
    mode= 'ms'
    solver_args = dict( solver='mszero', maxiter=4, fine_steps=4)

    t_span= torch.linspace(0,1,10)
    learner_args = dict(t_span=t_span)

    trainer_args = dict(max_epochs=3, enable_checkpointing=False, fast_dev_run=False)
    train_mnist_ode(mode, solver_args, learner_args, trainer_args, test=True)


