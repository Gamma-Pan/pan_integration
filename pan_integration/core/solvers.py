import torch
from torch import Tensor, tensor, nn, cos, pi, arange, hstack, cat, arccos
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
        self, num_coeff_per_dim, device=None, callback=None, delta=0.1, patience=30
    ):
        super().__init__()

        self.callback = callback
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_coeff_per_dim - 2
        self.delta = delta
        self.patience = patience

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.B_R = None

        self.t_cheb, self.PHI, self.DPHI = self.calc_independent(
            self.num_coeff_per_dim, self.num_points, device=self.device
        )

    @staticmethod
    def calc_independent(num_coeff_per_dim, num_points, device):
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

    @staticmethod
    def batched_call(f, t, y):
        # treat time as batch dim
        batch_sz, *dims, time_sz = y.shape

        t_b = (
            t.reshape(time_sz, 1, *[1 for _ in dims])
            .expand(time_sz, batch_sz, *dims)
            .reshape(-1, *dims)
        )

        y_b = y.permute(-1, *range(len(dims) + 1)).reshape(-1, *dims)
        f_b = f(t_b, y_b)
        # revert back to time last
        f_bb = f_b.reshape(time_sz, batch_sz, *dims).permute(
            *range(1, len(dims) + 2), 0
        )
        return f_bb

    def _fixed_point(self, f, t_lims, y_init, f_init, B_init):
        dims = y_init.shape
        dt = 2 / (t_lims[-1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[-1] - t_lims[0]) * (self.t_cheb[1:] + 1)

        B = self._add_head(B_init, dt, y_init, f_init)
        y_approx = B @ self.PHI
        f_approx = self.batched_call(
            f, t_true, y_approx[..., 1:]
        )  # -1 is constrained by b_head

        idx = 0
        prev_pointer = torch.tensor([0], device=self.device)
        while idx <= self.patience:
            B_R = linalg.solve_ex(
                self.DPHI[2:, 1:] - self.DPHI[2:, [0]],
                (1 / dt) * (f_approx - f_init[..., None]),
                left=False,
            )[0]
            B = self._add_head(B_R, dt, y_init, f_init)

            y_approx = B @ self.PHI
            f_approx = self.batched_call(
                f, t_true, y_approx[..., 1:]
            )  # -1 is constrained by b_head

            if self.callback is not None:
                # PASS EVERYTHING
                self.callback(
                    t_lims,
                    y_init.detach(),
                    f_init.detach(),
                    y_approx.detach(),
                    f_approx.detach(),
                    B.detach(),
                    self.PHI,
                    self.DPHI,
                )

            d_approx = B @ self.DPHI[..., 1:]
            mask = (
                linalg.vector_norm(
                    f_approx - dt * d_approx, dim=(*range(0, f_approx.dim() - 1),)
                )
                < self.delta
            )
            # if solution converges in this y_init
            if torch.all(mask):
                return y_approx[..., -1], f_approx[..., -1]

            pointer = mask.logical_not().nonzero()[0]
            if pointer > prev_pointer:
                idx = 0
                prev_pointer = torch.max(pointer, prev_pointer)
                continue
            prev_pointer = torch.max(pointer, prev_pointer)
            idx += 1

        if pointer > 0:
            new_lims = (t_true[pointer - 1], t_lims[1])
            y_init = y_approx[..., pointer[0]]
            f_init = f_approx[..., pointer[0]]
            return self._fixed_point(f, new_lims, y_init, f_init, B_R)
        else:
            print("ufck")
            # if it can even converge at the first point go deep
            new_lims = (t_lims[0], t_true[0])
            y0, f0 = self._fixed_point(f, new_lims, y_init, f_init, B_R)
            new_lims = (t_true[0], t_lims[1])
            return self._fixed_point(f, new_lims, y0, f0, B_R)

    def solve(self, f, t_span, y_init, f_init=None):

        y_eval = [y_init]

        if f_init is None:
            f_init = f(t_span[0], y_init)

        B_init = torch.rand(
            (*y_init.shape, self.num_coeff_per_dim - 2), device=self.device
        )

        for t_lims in zip(t_span, t_span[1:]):
            yk, fk = self._fixed_point(f, t_lims, y_init, f_init, B_init)
            y_init = yk
            f_init = fk
            y_eval.append(yk)

        return torch.stack(y_eval, dim=0)
