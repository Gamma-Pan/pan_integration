import torch
import torch.nn as nn
import matplotlib as mpl
import torchdyn as dyn
import lightning as lit

mpl.use("TkAgg")

device = torch.device("cuda:0")

# first define f (the neural net) and then the neural ODE module
vf = nn.Sequential(nn.Linear(2, 64),
                   nn.Tanh(),
                   nn.Linear(64, 2))

ode_model = dyn.core.NeuralODE(vf, sensitivity='adjoint')

class LitLearner(lit.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.t_span = torch.linspace(0,1,100)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X, return_t_eval = False)




def train():
    f = VectorField()
    t_span = torch.linspace(0, 1, 100)
    ode_model = NeuralODE(f, sensitivity="adjoint", solver="rk4")
    lit_module = LitLearner(t_span, ode_model)

    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=None,
        log_every_n_steps=10,
        # fast_dev_run=True
    )

    d = ToyDataset()
    num_samples = 10000
    X, y = d.generate(n_samples=num_samples, noise=2e-1, dataset_type="moons")
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    trainer.fit(lit_module, train_dataloaders=train_loader)

    with torch.no_grad():
        f.eval()
        x_lim = (2 * torch.min(X[:, 0]), 2 * torch.max(X[:, 0]))
        y_lim = (2 * torch.min(X[:, 1]), 2 * torch.max(X[:, 1]))
        plotter = plotting.VfPlotter(f, X[0], 0.,
                                     grid_definition=(40, 40), ax_kwargs={'xlim': x_lim, 'ylim': y_lim})
        plotter.ax.scatter(X[::100, 0], X[::100, 1], c=y[::100])
        for i in range(0, num_samples, 100):
            plotter.solve_ivp([0., 1.], X[i, :], ivp_kwargs={'method': "RK45"})

        plt.show()


if __name__ == '__main__':
    train()
