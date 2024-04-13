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


class LSZero(MultipleShootingDiffeqSolver):
    def __init__(self, num_coeff_per_dim, num_points, etol=1e-5, callback=None):
        super().__init__("euler", "euler")
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_points
        self.etol = etol
        self.callback = None

    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        y_init = x
        t_lims = [t_span[0], t_span[-1]]

        batches, dims = y_init.shape
        C = torch.rand(batches, self.num_coeff_per_dim - 2, dims)

        t = -torch.cos(torch.pi * (torch.arange(var_num_points) / var_num_points))
        d = t_lims[1] * torch.diff(torch.cat((t, tensor([1.0]))))[:, None]

        Phi, DPhi = _cheb_phis(self.num_points, self.num_coeff_per_dim, t_lims)

        inv0 = inv(stack((Phi[:, 0, [0, 1]], DPhi[:, 0, [0, 1]]), dim=1))
        Phi_aT = DPhi[:, :, [0, 1]] @ inv0 @ stack((y_init, f_init), dim=1)
        Phi_bT = (
            -DPhi[:, :, [0, 1]] @ inv0 @ cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1)
            + DPhi[:, :, 2:]
        )
        l = lambda C: inv0 @ (
            stack((y_init, f_init), dim=1)
            - cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1) @ C
        )

        Q = inv(Phi_bT.mT @ (d * Phi_bT))
        # MAIN LOOP
        for i in range(50):
            if self.callback is not None:
                self.callback(C.cpu())

            C_prev = C
            # C update
            C = Q @ (
                Phi_bT.mT @ (d * f(0, Phi @ cat((l(C), C), dim=1)))
                - Phi_bT.mT @ (d * Phi_aT)
            )

            if torch.norm(C - C_prev) < self.etol:
                break

        # refine with newton

        Phi_out, _ = _cheb_phis(
            len(t_span), self.num_coeff_per_dim, t_lims, include_end=True
        )

        approx = Phi_out @ cat((l(C), C), dim=1)
        # return (time,batch,dim)
        return approx.transpose(0, 1)


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

    t_lims = [0.0, 5.0]

    plotter = VfPlotter(f)
    plotter.solve_ivp(
        torch.linspace(*t_lims, 10), y_init, set_lims=True, plot_kwargs=dict(alpha=0.3)
    )

    def callback(B_vec):
        Phi, DPhi = _cheb_phis(
            num_coeff_per_dim=var_num_coeff_per_dim,
            num_points=var_num_points,
            t_lims=var_t_lims,
            include_end=True,
        )
        B = _B_init_cond(
            B_vec.reshape(
                2,
                var_num_coeff_per_dim - 2,
            ).T,
            var_y_init,
            f_init,
            Phi,
            DPhi,
        )
        approx = Phi @ B
        Dapprox = DPhi @ B
        plotter.approx(approx, var_t_lims[0], Dapprox=Dapprox)
        wait()

    num_coeff_per_dim = 7
    num_points = 7

    # solver = LSZero(var_num_coeff_per_dim, var_num_points)
    # _, approx_ms = odeint_mshooting(f, var_y_init, var_t_span, solver, torch.empty(0))

    B = lst_sq_solver(f, y_init, f_init, t_lims, num_coeff_per_dim, num_points)
    Phi, _ = _cheb_phis(num_points, num_coeff_per_dim, t_lims, include_end=True)
    approx = (Phi@B).transpose(0, 1)

    plotter.approx(approx, t_lims[0], color="orange")

    wait()
