import torch
from torch import nn
import torch.nn.functional as F

import pan_integration.utils.plotting
from pan_integration.core.pan_ode import PanODE, PanSolver, PanSolver
from torchdyn.models import NeuralODE
import torchdyn
from pan_integration.utils.plotting import wait

torch.manual_seed(28)

DIM = 16


class NN(nn.Module):
    def __init__(self, wstd=2):
        super().__init__()
        self.w1 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w1.weight, std=wstd)
        self.w2 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w2.weight, std=wstd)
        self.w3 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w3.weight, std=wstd)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = F.tanh(self.w1(x))
        x = F.tanh(self.w2(x))
        x = F.tanh(self.w3(x))
        return x


import matplotlib.pyplot as plt
from pan_integration.core.solvers import T_grid, DT_grid
from math import ceil, floor, sqrt

fig_fwd, ax_fwd = plt.subplots(4, 4)

y_init_fwd = None
y_init_bwd = None

lines_fwd = 16 * [None]
lines_bwd = 16 * [None]

coeff = 16
coeff_adj = 32
steps = 6


def callback_fwd(t_lims, y_init, B):
    global y_init_fwd
    global line_fwd

    t_plot = torch.linspace(*t_lims, 100)
    Phi = T_grid(torch.linspace(-1, 1, 100), coeff)
    approx = B @ Phi

    if y_init_fwd is not None and torch.allclose(y_init_fwd, y_init):
        for i, ax in enumerate(fig_fwd.axes):
            lines_fwd[i].set_data(t_plot, approx[0, i, :])
    else:
        y_init_fwd = y_init
        for i, ax in enumerate(fig_fwd.axes):
            lines_fwd[i] = ax.plot(t_plot, approx[0, 0, :], "r--")[0]
    fig_fwd.canvas.draw()
    fig_fwd.canvas.flush_events()


def callback_bwd(t_lims, y_init, B):
    global y_init_bwd
    global line_bwd

    t_plot = torch.linspace(*t_lims, 100)
    Phi = T_grid(torch.linspace(-1, 1, 100), coeff_adj)
    approx = B @ Phi

    if y_init_bwd is not None and torch.allclose(y_init_bwd, y_init):
        for i, ax in enumerate(fig_bwd.axes):
            lines_bwd[i].set_data(t_plot, approx[0, i, :])
    else:
        y_init_bwd = y_init
        for i, ax in enumerate(fig_bwd.axes):
            lines_bwd[i] = ax.plot(t_plot, approx[0, 0, :], "r--")[0]

    fig_bwd.canvas.draw()
    fig_bwd.canvas.flush_events()
    # wait()


if __name__ == "__main__":
    vf = NN(wstd=2)

    y_init = torch.rand(1, DIM)
    t_span = torch.linspace(0, 1, steps)

    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-4, atol_adjoint=1e-4
    )

    t_plot = torch.linspace(0, 1, 100)
    traj = ode_model(y_init, t_plot)

    with torch.no_grad():
        for idx ,ax  in enumerate(fig_fwd.axes):
            ax.plot(t_plot, traj[:, 0, idx].detach().numpy(), "b--")

    L = torch.sum((traj[-1] - 2 * torch.ones_like(y_init)) ** 2)
    L.backward()

    grads = [w.grad for w in vf.parameters()].copy()
    wait()

    vf.zero_grad()

    solver_conf = dict(num_coeff_per_dim=coeff, max_iters=50, delta=1e-4)
    solver_conf_adjoint = dict(num_coeff_per_dim=coeff_adj, max_iters=50, delta=1e-4)

    solver = PanSolver(**solver_conf, callback= None)# callback_fwd)
    solver_adjoint = PanSolver(**solver_conf_adjoint, callback=callback_bwd)

    pan_ode_model = PanODE(vf, t_span, solver, solver_adjoint, sensitivity="adjoint")

    _, traj_pan = pan_ode_model(y_init, t_span)

    L_pan = torch.sum((traj_pan[-1] - 2 * torch.ones_like(y_init)) ** 2)

    fig_bwd, ax_bwd = plt.subplots(4,4)
    L_pan.backward()

    wait()
    grads_pan = [w.grad for w in vf.parameters()].copy()

    print("-+=+-")

    print("SOLUTON \n")
    print(traj[-1])
    print(traj_pan[-1], "\n")

    print("GRADS\n")
    print(torch.norm(grads[0] - grads_pan[0]), "\n")
    print(torch.norm(grads[1] - grads_pan[1]), "\n")
    print(torch.norm(grads[2] - grads_pan[2]), "\n")
