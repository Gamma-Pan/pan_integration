from pan_integration.solvers.pan_integration import (
    lst_sq_solver,
    _cheb_phis,
    _B_init_cond,
)
import torch
from torch import nn


class Spiral(nn.Module):
    def __init__(self, a=0.1):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.linear.weight = nn.Parameter(
            torch.tensor([[-a, 1.0], [-1.0, -a]], requires_grad=True)
        )

    def forward(self, x):
        return self.linear(x)


f = Spiral()
f.requires_grad_(False)
f.eval()

if __name__ == "__main__":
    y_init = torch.rand(1, 2)
    f_init = f(y_init)

    t_lims = [0.0, 1.0]

    num_coeff_per_dim = 50
    num_points = 50
    B_vec, nfe = lst_sq_solver(
        f,
        y_init[0],
        f_init[0],
        [0.0, 1.0],
        num_coeff_per_dim=num_coeff_per_dim,
        num_points=num_points,
        return_nfe=True,
    )

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
    print(
        f"Method lstsq took {nfe}: \t\t\t\t function evaluations  y(T) = ({approx[-1,0].item()},{approx[-1,0].item()}"
    )


