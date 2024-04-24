import torch
from torch import nn, tensor
from torch.nn.functional import tanh, relu
from pan_integration.solvers.pan_integration import make_pan_adjoint
from lightning.pytorch import LightningModule, Trainer
from pan_integration.data import MNISTDataModule
from torchdyn.core import MultipleShootingLayer
from torchdyn.numerics.solvers.ode import MultipleShootingDiffeqSolver

torch.manual_seed(42)


class Learner(LightningModule):
    def __init__(
        self, model: nn.Module, t_span: torch.Tensor = torch.linspace(0, 1, 10)
    ):
        super().__init__()
        self.t_span, self.model = t_span, model
        self.linear = nn.Linear(28 * 28, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x.reshape(-1, 28 * 28), self.t_span)
        logits = self.linear(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        nfe = self.model.vf.nfe

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x.reshape(-1, 28 * 28), self.t_span)
        logits = self.linear(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        _, preds = torch.max(logits, dim=1)
        acc = torch.sum(preds == y) / y.shape[0]

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0005)


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
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(28 * 28, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 28 * 28)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = relu(self.lin1(x))
        x = relu(self.lin2(x))
        x = relu(self.lin3(x))
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_init = torch.rand(10, 2).requires_grad_(True).to(device)

    vf = VF().to(device)
    t_span = torch.linspace(0, 1, 50).to(device)

    # model = NeuralODE( vf, sensitivity="adjoint", solver="tsit5", atol=1e-6, rtol=1e-6, atol_adjoint=1e-6, rtol_adjoint=1e-6, ).to(device)

    # model = PanODE(
    #     vf,
    #     num_coeff_per_dim=30,
    #     num_points=50,
    #     # num_coeff_per_dim_adjoint=50,
    #     # num_points_adjoint=80,
    #     etol_ls=1e-3,
    # )


    model = MultipleShootingLayer(
        vf,
        solver="ieuler",
        # solver_adjoint ='tsit5'
        # return_t_eval=False
    ).to(device)

    learner = Learner(model)
    trainer = Trainer(
        max_epochs=5,
        # callbacks=[
        #     EarlyStopping(
        #         monitor="val_acc", stopping_threshold=0.9, mode="max", patience=5
        #     )
        # ],
    )
    dmodule = MNISTDataModule(batch_size=16, num_workers=12)
    trainer.fit(learner, datamodule=dmodule)
