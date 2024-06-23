import torch
from torch import Tensor, tensor, nn, cos, pi, arange, hstack, cat, arccos, linspace
from torch.func import vmap
from torch import linalg
from typing import Tuple, Callable

stackcounter = 0


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
        device=None,
        callback=None,
        delta=0.1,
        patience=30,
        max_iters=100,
    ):
        super().__init__()

        self.callback = callback
        self._num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_coeff_per_dim - 2
        self.delta = delta
        self.patience = patience
        self.max_iters = max_iters

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.t_cheb, self.PHI, self.DPHI, self.DPHI_inv = self.calc_independent(
            self.num_coeff_per_dim, self._num_coeff_per_dim - 2, device=self.device
        )

    @property
    def num_coeff_per_dim(self):
        return self._num_coeff_per_dim

    @num_coeff_per_dim.setter
    def num_coeff_per_dim(self, value):
        self.num_points = value - 2
        self.t_cheb, self.PHI, self.DPHI, self.DPHI_inv = self.calc_independent(
            value, value - 2, device=self.device
        )
        self._num_coeff_per_dim = value

    @staticmethod
    def calc_independent(num_coeff_per_dim, num_points, device):
        N = num_points
        # chebyshev nodes of the second kind
        k = arange(0, N + 1)
        t_cheb = -cos(pi * k / N).to(device)

        PHI = T_grid(t_cheb, num_coeff_per_dim)
        DPHI = DT_grid(t_cheb, num_coeff_per_dim)
        DPHI_inv = linalg.inv(DPHI[2:, 1:] - DPHI[2:, [0]])

        return t_cheb, PHI, DPHI, DPHI_inv

    def _add_head(self, B_R, dt, y_init, f_init):
        Phi_R0 = self.PHI[2:, [0]]
        DPhi_R0 = self.DPHI[2:, [0]]

        phi_inv = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=self.device)
        b_01 = (
            torch.stack([y_init, (1 / dt) * f_init], dim=-1)
            - B_R @ torch.hstack([Phi_R0, DPhi_R0])
        ) @ phi_inv

        return torch.cat([b_01, B_R], dim=-1)

    def _fixed_point(self, f, t_lims_init: Tuple, y_init: Tensor, f_init:Tensor ):
        dims = y_init.shape
        y_init = y_init.view(-1)
        all_t_lims = [t_lims_init]

        patience = 0
        idx = 0
        prev_pointer = torch.tensor([0], device=self.device)
        solve_flag = False

        for t_lims in all_t_lims:
            dt = 2 / (t_lims[1] - t_lims[0])
            t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_cheb[1:] + 1)

            while idx < self.max_iters:
                idx += 1

                B_R = ((1 / dt) * (f_approx - f_init[..., None])) @ self.DPHI_inv
                B = self._add_head(B_R, dt, y_init, f_init)

                y_approx = B @ self.PHI
                f_approx = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                    t_true,
                    y_approx[..., 1:].reshape(*dims, -1),  # t=-1 is constrained by b0
                ).reshape(-1, self.num_points)

                d_approx = dt * B @ self.DPHI[..., 1:]

                if not solve_flag:
                    patience += 1
                    pointer = -1

                    cos_sim = torch.sum(
                        (f_approx / f_approx.norm(dim=0))
                        * (d_approx / d_approx.norm(dim=0)),
                        dim=0,
                    )

                    if self.callback is not None:
                        self.callback(
                            t_lims,
                            y_init.detach().reshape(*dims),
                            f_init.detach().reshape(*dims),
                            B.detach().reshape(*dims, self.num_coeff_per_dim),
                        )

                    if (cos_sim >= 0.8).all() or patience > self.patience:
                        solve_flag = True
                        patience = 0
                        prev_pointer = tensor([0], device=self.device)
                        t_lims.append( [t_lims[0], t_true[pointer-1]])

                        continue

                    print(
                        f"total: {idx} | prev_pointer: {prev_pointer.item()} -> pointer: {pointer.item()} | patience: {patience} / {self.patience} , tlims={t_lims}"
                    )

                    if (pointer := (cos_sim < 0.8).nonzero()[0]) > prev_pointer:
                        prev_pointer = pointer
                        patience = 0
                        continue

                if solve_flag:
                    if self.callback is not None:
                        self.callback(
                            t_lims,
                            y_init.detach().reshape(*dims),
                            f_init.detach().reshape(*dims),
                            B.detach().reshape(*dims, self.num_coeff_per_dim),
                        )

                    print(
                        f"total: {idx} |  patience: {patience} / {self.patience} "
                        f"| delta {(d_approx - f_approx).norm()}| tlims={t_lims}"
                    )

                    patience += 1

                    # solve in specified interval

                    if (d_approx - f_approx).norm() < self.delta:
                        patience = 0
                        y_init = y_approx[:, -1].squeeze()
                        f_init = f_approx[:, -1].squeeze()

                        solve_flag = False
                        t_lims = [t_split, t_lims_init[1]]
                        dt = 2 / (t_lims[1] - t_lims[0])
                        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (
                            self.t_cheb[1:] + 1
                        )

    def solve(self, f, t_span, y_init, f_init=None, B_init=None):

        if f_init is None:
            f_init = f(t_span[0], y_init).reshape(-1)

        y_eval = [y_init]
        f_eval = [f_init]

        for t_lims in zip(t_span, t_span[1:]):
            yk, fk, Bk = self._fixed_point(f, t_lims, y_init, f_init, B_init)
            y_init = yk
            f_init = fk
            B_init = Bk
            y_eval.append(yk)
            f_eval.append(fk)

        return torch.stack(y_eval, dim=0), torch.stack(f_eval, dim=0)
