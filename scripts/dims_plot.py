import torch
from torch import tensor, nn, linalg, cos, sin, cosh, exp
from torch.nn import functional as F
from torch.func import jacrev
from pan_integration.utils.plotting import wait, DimPlotter

from scipy.optimize import linprog
from torchdiffeq import odeint


import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("TkAgg")
# torch.set_default_dtype(torch.float64)

torch.manual_seed(23)
from torch.linalg import inv
from pan_integration.utils.plotting import wait
from pan_integration.core.solvers import T_grid, DT_grid, PanSolver


torch.manual_seed(68)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DIMS = 1
NUM_COFF = 20
INTERS = 20

a = 1000

class NN(nn.Module):
    def __init__(self, std=5.0, hidden_dim=5):
        super().__init__()
        self.w1 = torch.nn.Linear(DIMS, hidden_dim)
        torch.nn.init.normal_(self.w1.weight, std=std)
        self.w2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.normal_(self.w2.weight, std=std)
        self.w3 = torch.nn.Linear(hidden_dim, DIMS)
        torch.nn.init.normal_(self.w3.weight, std=std)
        # self.A = torch.tensor([[-0.9, -2.0], [1.5, -1]], device=device)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        y = torch.tanh(self.w1(y))
        y = F.softplus(self.w2(y))
        y = torch.tanh(self.w3(y))
        return y


class Fun(nn.Module):
    def __init__(self, c=0.1):
        super().__init__()
        self.nfe = 0
        self.c = c

    def forward(self, t, y):
        self.nfe += 1
        return -a*y + a*sin(t)


if __name__ == "__main__":

    # f = NN(std=4).to(device)
    f = Fun(c=0.5).to(device)
    for param in f.parameters():
        param.requires_grad_(False)

    y_init = torch.tensor([[1.]], device=device)
    # y_init = torch.randn((1, DIMS), device=device)
    t_lims = tensor([0., 1.], device=device)
    t_span = torch.linspace(*t_lims, 2, device=device)

    plotter = DimPlotter(
        f,
        [[0,i] for i in range(DIMS)],
    )

    coarse_dopri = plotter.solve_ivp(
        t_span,
        y_init,
        ivp_kwargs=dict(solver="dopri5", atol=1e-4, rtol=1e-4),
        plot_kwargs=dict(color="blue", alpha=0.1, label='coarse_dopri'),
    )

    fine_dopri = plotter.solve_ivp(
        t_span,
        y_init,
        ivp_kwargs=dict(solver="dopri5", atol=1e-8, rtol=1e-8),
        plot_kwargs=dict(color="green", alpha=0.4, label='fine_dopri'),
    )

    true_t = torch.linspace(*t_lims, 100)
    c = 1 + 1/a
    true_sol = sin(true_t) - (1/a)*cos(true_t) + c*exp(-a*true_t)
    plotter.axes.plot(true_t,true_sol, 'y--', label='true_sol')
    plotter.axes.legend()

    # print(f"dopri5 sol {torch.norm(exact_sol[-1] - tsit_sol[-1])}")
    wait()

    f.nfe = 0
    approx_text = plotter.fig.text(x=0.6, y=0.05, s=f"nfe = {f.nfe}", color="black")

    def callback(i, t_lims, y_init, f_init, B, **kwargs):
        approx = plotter.approx(
            tensor(t_lims),
            B,
            num_arrows=0,
            num_points=100,
            marker='o',
            markersize=2.5,
            alpha=0.70,
            color="red",
        )


        if not i % 10:
            approx_text.set_text(f"nfe = {f.nfe} ")
            plotter.fig.canvas.flush_events()
            plotter.fig.canvas.draw()
            # wait()


        #stop?
        return False

    solver = PanSolver(
        num_coeff_per_dim=NUM_COFF,
        callback=callback,
        device=device,
        tol=1e-2,
        max_iters=100_000,
    )

    approx, _, B_ch = solver.solve(
        f,
        torch.linspace(*t_lims, INTERS+1),
        y_init,
    )

    plt.show()
    #########################################################################



