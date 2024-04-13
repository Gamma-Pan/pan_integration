from pan_integration.solvers.pan_integration import (
    lst_sq_solver,
    _cheb_phis,
    _B_init_cond,
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
    batches = 1
    dims = 2
    y_init = torch.rand(batches, dims)
    f_init = f(0, y_init)


    # t_eval, y_tsit = odeint(f, y_init, t_span, "tsit5")

    def callback(B_vec):
        Phi, DPhi = _cheb_phis(
            num_coeff_per_dim=num_coeff_per_dim,
            num_points=num_points,
            t_lims=t_lims,
            include_end=True,
        )
        B = _B_init_cond(
            B_vec.reshape(
                2,
                num_coeff_per_dim - 2,
            ).T,
            y_init,
            f_init,
            Phi,
            DPhi,
        )
        approx = Phi @ B
        Dapprox = DPhi @ B
        plotter.approx(approx, t_lims[0], Dapprox=Dapprox)
        wait()

    # solver = LSZero(var_num_coeff_per_dim, var_num_points)
    # _, approx_ms = odeint_mshooting(f, var_y_init, var_t_span, solver, torch.empty(0))

    t_lims = [0.0, 5.0]
    num_points = 7
    num_coeff_per_dim = 7

    B = lst_sq_solver(f,t_lims, y_init,num_coeff_per_dim,num_points, f_init=f_init)

    t_span = -torch.cos(torch.pi * (torch.arange(num_points+1) / num_points))
    Phi, _ = _cheb_phis(t_span, 1 , num_coeff_per_dim)
    approx = (Phi @ B).transpose(0, 1)


    plotter = VfPlotter(f)
    plotter.solve_ivp(torch.linspace(*t_lims, steps=100), y_init, set_lims=True,plot_kwargs=dict(alpha=0.3))
    wait()
    plotter.approx(approx, t_lims[0], color="orange")
    wait()
