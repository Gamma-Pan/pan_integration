from pan_integration.utils.plotting import VfPlotter, wait
from pan_integration.solvers.pan_integration import (
    pan_int,
    newton_solver,
    T_grid,
    U_grid,
    lst_sq_solver,
)

import torch
from torch import nn, tensor, cat, stack
from torch.linalg import inv

from torchdyn.numerics import odeint, odeint_mshooting
from torchdyn.numerics.solvers.templates import MultipleShootingDiffeqSolver

torch.manual_seed(42)


class Spiral(nn.Module):
    def __init__(self, dims, a=0.3):
        super().__init__()
        # self.linear = nn.Linear(2, 2)
        # self.linear.weight = nn.Parameter(
        #     torch.tensor([[-a, 1.0], [-1.0, -a]], requires_grad=True)
        # )
        # self.linear.bias = nn.Parameter(torch.zeros((1, 2), requires_grad=False))
        self.linear = nn.Sequential(
            nn.Linear(dims, 512), nn.Tanh(), nn.Linear(512,512), nn.Tanh(),nn.Linear(512, dims), nn.Tanh()
        )
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.linear(torch.tanh(self.linear(x) + t[..., None]**2))


if __name__ == "__main__":
    batches = 32
    dims = 10

    device = torch.device("cpu")
    # device = torch.device("cuda")

    f = Spiral(dims).to(device)
    f.requires_grad_(False)
    f.eval()

    y_init = torch.rand(batches, dims).to(device)
    t_lims = [0.0, 1.0]

    num_points = 7
    num_coeff_per_dim = 7

    t_span = torch.linspace(t_lims[0], t_lims[1], num_points, device=device)
    t_phi = (-1 + 2 * (t_span - t_lims[0]) / (t_lims[1] - t_lims[0]) ).to(device)
    Phi_traj = T_grid(t_phi, num_coeff_per_dim, device)

    B_init = torch.rand(batches, num_coeff_per_dim - 2, dims, device=device)

    f.nfe = 0
    print(10 * "-" + "\n" + "tsit5")
    t_eval, sol_tsit = odeint(f, y_init, t_span, solver="tsit5", atol=1e-5, rtol=1e-5)
    print(f.nfe)
    print(sol_tsit[-1][-1])

    f.nfe = 0
    print(10 * "-" + "\n" + "pan")
    sol_pan = pan_int(
        f, t_span, y_init, num_coeff_per_dim, num_points, etol_ls=1e-5, etol_newton=1e-7
    )
    print(f.nfe)
    print(sol_pan[-1][-1])

    print('------')
    print('pan', torch.norm(sol_pan[-1] - sol_tsit[-1]))

    plotter = VfPlotter(f)
    # plotter.solve_ivp(
    #     torch.linspace(*t_lims, steps=100),
    #     y_init,
    #     set_lims=True,
    #     plot_kwargs=dict(alpha=0.3),
    # )
    # wait()
    # plotter.approx(trajectory, t_lims[0], color="orange")
    # wait()
