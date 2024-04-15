from pan_integration.solvers.pan_integration import (
    lst_sq_solver,
    newton_solver,
    _B_init_cond,
    T_grid,
)

from pan_integration.utils.plotting import VfPlotter, wait

import torch
from torch import nn, tensor, cat, stack
from torch.linalg import inv

from torchdyn.numerics import odeint, odeint_mshooting
from torchdyn.numerics.solvers.templates import MultipleShootingDiffeqSolver

torch.manual_seed(42)


class Spiral(nn.Module):
    def __init__(self, a=0.3):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.linear.weight = nn.Parameter(
            torch.tensor([[-a, 1.0], [-1.0, -a]], requires_grad=True)
        )
        self.linear.bias = nn.Parameter(torch.zeros((1, 2), requires_grad=False))

    def forward(self, t, x):
        return self.linear(x)


f = Spiral()
f.requires_grad_(False)
f.eval()

if __name__ == "__main__":
    batches = 5
    dims = 2
    y_init = torch.rand(batches, dims)
    f_init = f(0, y_init)

    t_lims = [0.0, 5.0]
    num_points = 10
    num_coeff_per_dim = 10

    def callback(B_vec):
        t_span = -1 + 2 * torch.linspace(*t_lims, steps=100) / (t_lims[1] - t_lims[0])
        Phi = T_grid(t_span, num_coeff_per_dim)
        approx = (Phi @ B).transpose(0, 1)
        plotter.approx(approx, t_lims[0])
        wait()

    # B = lst_sq_solver(f, t_lims, y_init, num_coeff_per_dim, num_points, f_init=f_init)
    B_init = torch.rand(batches, num_coeff_per_dim - 2, dims).reshape(batches, -1)
    B = newton_solver(
        f,
        t_lims,
        y_init,
        num_coeff_per_dim,
        num_points,
        f_init=f_init,
        B_init=B_init,
    )

    t_span = -1 + 2 * torch.linspace(*t_lims, steps=100) / (t_lims[1] - t_lims[0])
    Phi = T_grid(t_span, num_coeff_per_dim)
    approx = (Phi @ B).transpose(0, 1)

    plotter = VfPlotter(f)
    plotter.solve_ivp(torch.linspace(*t_lims, steps=100), y_init, set_lims=True, plot_kwargs=dict(alpha=0.3))
    wait()
    plotter.approx(approx, t_lims[0], color="orange")
    wait()
