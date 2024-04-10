from pan_integration.solvers.pan_integration import (
    lst_sq_solver,
    _cheb_phis,
    _B_init_cond,
)

from pan_integration.utils.plotting import VfPlotter, wait

import torch
from torch import nn, tensor
import scipy

from torchdyn.numerics import odeint

torch.manual_seed(42)


class Spiral(nn.Module):
    def __init__(self, a=0.3):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.linear.weight = nn.Parameter(
            torch.tensor([[-a, 1.0], [-1.0, -a]], requires_grad=True)
        )
        self.linear.bias = nn.Parameter(torch.zeros((1, 2), requires_grad=False))

    def forward(self, x):
        return self.linear(x)


f = Spiral()
f.requires_grad_(False)
f.eval()

if __name__ == "__main__":
    batches = 3
    dims=2
    y_init = torch.rand(batches,dims)
    f_init = f(y_init)

    t_lims = [0., 4.]
    t_span = torch.linspace(*t_lims, 10)

    plotter = VfPlotter(f, y_init[0], t_lims[0])
    plotter.solve_ivp(t_lims, y_init[0], set_lims=True)

    t_eval, y_tsit = odeint(lambda t, x:f(x),y_init,t_span, 'tsit5' )

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

    num_coeff_per_dim =  8
    num_points = 10
    B_vec, nfe = lst_sq_solver(
        f,
        y_init,
        f_init,
        t_lims,
        num_coeff_per_dim=num_coeff_per_dim,
        num_points=num_points,
        return_nfe=True,
        etol=1e-9,
        callback=None
    )

    Phi, DPhi = _cheb_phis(num_points, num_coeff_per_dim,t_lims,include_end=True)
    B_tail = B_vec.reshape(batches, dims, num_coeff_per_dim - 2 ).mT
    B = _B_init_cond(B_tail, y_init, f_init, Phi, DPhi)
    approx = Phi@B
    plotter.approx(approx[0,...], t_lims[0])
    wait()
