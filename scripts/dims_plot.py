import torch
from torch import tensor, nn, linalg
from torch.nn import functional as F
from torch.optim import Adam, SGD, Adadelta

from pan_integration.core.pan_ode import PanSolver, T_grid
from pan_integration.utils.plotting import wait, DimPlotter

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter

from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint


# torch.manual_seed(23)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DIMS = 16


class NN(nn.Module):
    def __init__(self, std=5.0):
        super().__init__()
        self.w1 = torch.nn.Linear(DIMS, 10)
        torch.nn.init.normal_(self.w1.weight, std=std)
        self.w2 = torch.nn.Linear(10, 10)
        torch.nn.init.normal_(self.w2.weight, std=std)
        self.w3 = torch.nn.Linear(10, DIMS)
        torch.nn.init.normal_(self.w3.weight, std=std)
        self.A = torch.tensor([[-0.9, -2.0], [1.5, -1]], device=device)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        y = torch.cos(0.4 * self.w1(y))
        y = torch.cos(0.4 * self.w2(y))
        y = torch.cos(0.4 * self.w3(y))
        return y


class EXP(nn.Module):
    def __init__(self, c=0.1):
        super().__init__()
        self.nfe = 0
        self.c = c

    def forward(self, t, y):
        self.nfe += 1
        return self.c * torch.cos(y)


if __name__ == "__main__":

    f = NN(std=3).to(device)
    # f = EXP(c=2.).to(device)
    for param in f.parameters():
        param.requires_grad_(False)

    # y_init = torch.tensor([[0.], [0.5], [-0.5]], device=device)
    y_init = torch.randn((16, DIMS), device=device)
    t_lims = tensor([0.0, 1.0], device=device)
    t_plot = torch.linspace(*t_lims, 100)

    plotter = DimPlotter(
        f,
        [
            [0, 0],
            [0, 2],
            [1, 4],
            [1, 2],
            [2, 3],
            [2, 5],
            [3, 1],
            [3, 2],
            [4, 3],
            [5, 1],
            [5, 1],
            [6, 2],
            [7, 0],
            [7, 1],
            [8, 4],
            [8, 1],
        ],
    )
    plotter.solve_ivp(t_plot, y_init, plot_kwargs=dict(color="green", alpha=0.2))
    plotter.solve_ivp(
        t_plot,
        y_init,
        ivp_kwargs=dict(solver="dopri5", atol=1e-4, rtol=1e-4),
        plot_kwargs=dict(color="blue"),
    )
    wait()

    f.nfe = 0
    approx_text = plotter.fig.text(x=0.6, y=0.05, s=f"nfe = {f.nfe}", color="black")

    def callback(i, t_lims, y_init, f_init, B):
        approx = plotter.approx(
            tensor(t_lims),
            B,
            num_arrows=0,
            num_points=101,
            marker=None,
            markersize=2.5,
            alpha=0.70,
            color="red",
        )

        approx_text.set_text(f"nfe = {f.nfe} ")
        plotter.fig.canvas.flush_events()
        plotter.fig.canvas.draw()
        # wait()

    solver = PanSolver(
        num_coeff_per_dim=10,
        callback=callback,
        device=device,
        tol=1e-5,
        min_lr=0.001,
        max_iters=20,
        gamma=0.98,
    )

    optim_kwargs = dict(
        optim_class=Adam,
        lr=1e-3,
    )

    approx, _ = solver.solve(
        f,
        torch.linspace(*t_lims, 20, device=device),
        y_init,
        mode="gd",
        solver_kwargs=optim_kwargs,
    )

    plt.show()
