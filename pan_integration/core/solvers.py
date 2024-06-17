import torch
from torch import Tensor, tensor, nn, cos, pi, arange, hstack, cat
from torch.func import vmap
from torch import linalg
from typing import Tuple, Callable
from abc import ABC


def T_grid(t, num_coeff_per_dim):
    num_points = len(t)
    out = torch.empty(num_coeff_per_dim, num_points, device=t.device)

    out[0, :] = torch.ones(num_points)
    out[1, :] = t

    for i in range(2, num_coeff_per_dim):
        out[i, :] = 2 * t * out[i - 1, :] - out[i - 2, :]

    return out


def U_grid(t, num_coeff_per_dim):
    num_points = len(t)
    out = torch.empty(num_coeff_per_dim, num_points, device=t.device)

    out[0, :] = torch.ones(num_points)
    out[1, :] = 2 * t

    for i in range(2, num_coeff_per_dim):
        out[i, :] = 2 * t * out[i - 1, :] - out[i - 2, :]

    return out


def DT_grid(t, num_coeff_per_dim):
    num_points = len(t)

    out = torch.vstack(
        [
            torch.zeros(1, num_points, dtype=torch.float, device=t.device),
            U_grid(t, num_coeff_per_dim - 1)
            * torch.arange(1, num_coeff_per_dim, dtype=torch.float, device=t.device)[
                :, None
            ],
        ]
    )

    return out


class PanSolver:
    def __init__(
        self,
        num_coeff_per_dim,
        # num_points,
        delta=1e-2,
        max_iters=20,
        device=None,
        callback=None,
        t_span=None
    ):
        super().__init__()

        self.callback = callback
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_coeff_per_dim - 2
        self.delta = delta
        self.max_iters = max_iters
        self.t_span = t_span

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.B_R = None

        self.t_cheb, self.PHI, self.DPHI = self._calc_independent(
            self.num_coeff_per_dim, self.num_points, device=self.device
        )

    @staticmethod
    def _calc_independent(num_coeff_per_dim, num_points, device):
        N = num_points
        # chebyshev nodes of the second kind
        k = arange(0, N + 1)
        t_cheb = -cos(pi * k / N).to(device)

        PHI = T_grid(t_cheb, num_coeff_per_dim)
        DPHI = DT_grid(t_cheb, num_coeff_per_dim)

        return t_cheb, PHI, DPHI

    def _add_head(self, B_R, dt, y_init, f_init):
        Phi_R0 = self.PHI[2:, [0]]
        DPhi_R0 = self.DPHI[2:, [0]]

        phi_inv = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=self.device)
        b_01 = (
            torch.stack([y_init, (1 / dt) * f_init], dim=-1)
            - B_R @ torch.hstack([Phi_R0, DPhi_R0])
        ) @ phi_inv

        return torch.cat([b_01, B_R], dim=-1)

    def _fixed_point(self, f, t_lims, y_init, f_init):
        dt = 2 / (t_lims[-1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[-1] - t_lims[0]) * (self.t_cheb[1:] + 1)

        B = self._add_head(self.B_R, dt, y_init, f_init)

        y_approx = B @ self.PHI[..., 1:]  # don't care about -1
        f_approx = vmap(f, in_dims=(0, -1), out_dims=(-1,))(t_true, y_approx)

        for i in range(self.max_iters):
            prev_sol = y_approx[..., -1]
            # num_points != num_coeff - 2
            # B_R = linalg.lstsq(
            #     (self.DPHI[2:, 1:] - self.DPHI[2:, [0]])
            #     .expand(*(len(dims) - 1) * [1], self.num_coeff_per_dim - 2, self.num_points)
            #     .mT,
            #     ((1 / dt) * fapprox - f_init[..., None]).mT,
            # ).solution.mT

            self.B_R = linalg.solve_ex(
                self.DPHI[2:, 1:] - self.DPHI[2:, [0]],
                (1 / dt) * (f_approx - f_init[..., None]),
                left=False,
            )[0]

            B = self._add_head(self.B_R, dt, y_init, f_init)

            if self.callback is not None:
                self.callback(t_lims, y_init.detach(), B.detach())

            y_approx = B @ self.PHI[..., 1:]  # don't care about 0
            f_approx = vmap(f, in_dims=(0, -1), out_dims=(-1,))(t_true, y_approx)

            if torch.norm(y_approx[..., -1] - prev_sol) < self.delta:
                break

        return y_approx[..., -1], f_approx[...,-1]

    def solve(self, f, t_span, y_init, f_init=None):
        dims = y_init.shape

        if self.t_span is not None:
            t_span = self.tspan

        if f_init is None:
            f_init = f(t_span[0], y_init)

        if self.B_R is None or y_init.shape != self.B_R.shape:
            self.B_R = torch.randn((*dims, self.num_coeff_per_dim - 2), device=self.device)

        solution = [y_init]
        for t_lims in zip(t_span, t_span[1:]):
            yk, fk = self._fixed_point(f, t_lims, y_init, f_init)
            y_init = yk
            f_init = fk
            solution.append(yk)

        return torch.stack(solution, dim=0)
