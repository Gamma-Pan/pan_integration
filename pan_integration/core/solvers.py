import torch
from torch import Tensor, tensor, nn, cos, pi, arange, hstack, cat, arccos, linspace
from torch.func import vmap
from torch import linalg
from typing import Tuple, Callable
from functools import partial

import matplotlib.pyplot as plt

from torch.linalg import inv

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
        tol=0.01,
        min_lr=1e-3,
        patience=3,
        max_iters=100,
        gamma=0.9,
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.callback = callback
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_coeff_per_dim - 2

        self.tol = tol
        self.patience = patience
        self.gamma = gamma
        self.min_lr = min_lr

        self.max_iters = max_iters

    @property
    def num_coeff_per_dim(self):
        return self._num_coeff_per_dim

    @num_coeff_per_dim.setter
    def num_coeff_per_dim(self, value):
        self.num_points = value - 2
        (
            self.t_cheb,
            self.PHI,
            self.DPHI,
            self.PHI_r,
            self.DPHI_r,
            self.PHI_r_inv,
            self.DPHI_r_inv,
        ) = self.calc_independent(value, device=self.device)
        self._num_coeff_per_dim = value

    @staticmethod
    def calc_independent(num_coeff_per_dim, device):
        N = num_coeff_per_dim - 1
        # chebyshev nodes of the second kind
        k = arange(0, N)
        t_cheb = -cos(pi * k / (N - 1)).to(device)

        PHI = T_grid(t_cheb, num_coeff_per_dim)
        DPHI = DT_grid(t_cheb, num_coeff_per_dim)

        DPHI_r = DPHI[2:, 1:] - DPHI[2:, [0]]
        DPHI_r_inv = linalg.inv(DPHI_r)

        PHI_r = (
            -PHI[2:, [0]]
            - DPHI[2:, [0]]
            - DPHI[2:, [0]] * t_cheb[None, 1:]
            + PHI[2:, 1:]
        )
        PHI_r_inv = linalg.inv(PHI_r)

        return t_cheb, PHI, DPHI, PHI_r, DPHI_r, PHI_r_inv, DPHI_r_inv

    def _add_head(self, B_R, dt, y_init, f_init):
        Phi_R0 = self.PHI[2:, [0]]
        DPhi_R0 = self.DPHI[2:, [0]]

        phi_inv = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=self.device)
        b_01 = (
            torch.stack([y_init, (1 / dt) * f_init], dim=-1)
            - B_R @ torch.hstack([Phi_R0, DPhi_R0])
        ) @ phi_inv

        return torch.cat([b_01, B_R], dim=-1)

    def _B_rest(self, pointer, t_true, y_init, f_init, B_R, C, y_k, f_k):

        t_pointer = self.t_cheb[pointer]
        t_pcheb = t_pointer + 0.5 * (1 - t_pointer) * (self.t_cheb + 1)

        # loops! potential bottleneck (try cos(arccos) definition)
        PHI_old = T_grid(t_pcheb, self._num_coeff_per_dim)

        PHI_r_old = (
            -self.PHI[2:, [0]]
            - self.DPHI[2:, [0]]
            - self.DPHI[2:, [0]] * t_pcheb[None, 1:]
            + PHI_old[2:, 1:]
        )

        a = (t_true[-1] - t_true[0]) / 2
        C_old = a * f_init + y_init + a * f_init * t_pcheb[None, 1:]
        approx = C_old + B_R @ PHI_r_old

        ax = plt.gca()
        # ax.plot(approx[0, 0, :], approx[0, 1, :], color="yellow")
        # plt.pause(2)

        t_lims = [t_pointer, t_true[-1]]
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_cheb + 1)
        a = 0.5 * (t_lims[-1] - t_lims[0])
        y_init = y_k[..., [pointer - 1]]
        f_init = f_k[..., [pointer - 1]]
        C_new = a * f_init + y_init + a * f_init * self.t_cheb[None, 1:]
        B_R = (approx - C_new) @ self.PHI_r_inv

        approx_new = C_new + B_R @ self.PHI_r
        ax.plot(approx_new[0, 0, :], approx_new[0, 1, :], color="red")
        plt.pause(1)

        return t_lims, t_true, a, y_init, f_init, C_new, B_R

    def _zero_order(
        self, f, t_lims: Tuple, y_init: Tensor, f_init: Tensor, B_R: Tensor = None
    ):
        dims = y_init.shape
        a = 0.5 * (t_lims[1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_cheb + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]

        C = a * f_init + y_init + a * f_init * self.t_cheb[None, 1:]

        if B_R is None:
            B_R = torch.rand(*dims, self.num_coeff_per_dim - 2, device=self.device)

        lr = 0.99*torch.ones(*dims, device=self.device)
        gamma = torch.tensor(self.gamma, device=self.device)
        prev_rel_error = torch.inf * torch.ones(*dims, device=self.device)

        for i in range(self.max_iters):
            y_k = C + B_R @ self.PHI_r
            Dy_k = B_R @ self.DPHI_r
            f_k = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                t_true[1:],
                y_k,
            )

            DB_R = (a * f_k - a * f_init - Dy_k) @ self.DPHI_r_inv
            B_R = B_R + lr[...,None] * DB_R

            # error per sample
            rel_err = linalg.vector_norm(
                Dy_k - a * (f_k - f_init), dim=(-1,)
            )

            lr = lr * (
                (
                    1
                    + (1 - torch.heaviside(rel_err - prev_rel_error, torch.tensor(1.0)))
                    * (gamma - 1)
                )
            )

            prev_rel_error = rel_err

            if torch.norm(rel_err) < self.tol:
                break

            if self.callback is not None:
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                self.callback(i, t_lims, y_init, f_init, B)

        return y_k[..., -1], f_k[..., -1], B_R

    def _first_order(
        self, f, t_lims: Tuple, y_init: Tensor, f_init: Tensor, B_R: Tensor = None
    ):
        dims = y_init.shape
        a = 0.5 * (t_lims[1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_cheb + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]

        C = a * f_init + y_init + a * f_init * self.t_cheb[None, 1:]

        if B_R is None:
            B_R = torch.rand(*dims, self.num_coeff_per_dim - 2, device=self.device)

        lr = torch.tensor(1e-6, device=self.device)

        def single_f(t, y, dy):
            f_k, vjp_func = torch.func.vjp(f, t, y)
            f_k = a * (f_k - f_init[..., 0])
            err = dy - f_k
            _, vJp = vjp_func(dy - f_k)
            return err, vJp

        for i in range(self.max_iters):
            y_k = C + B_R @ self.PHI_r
            Dy_k = B_R @ self.DPHI_r

            err_k, vJp_k = vmap(
                single_f, in_dims=(0, -1, -1), out_dims=(-1, -1), randomness="same"
            )(t_true[1:], y_k, Dy_k)

            err = torch.trace(err_k + err_k.mT)
            grad = self.DPHI_r.T - vJp_k @ self.PHI_r.T

            B_R = B_R - lr * grad

            if self.callback is not None:
                if i > 3:
                    break
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                self.callback(i, t_lims, y_init, f_init, B)

        return y_k[..., -1], B_R

    def _gd(
        self,
        f,
        t_lims: Tuple,
        y_init: Tensor,
        f_init: Tensor,
        B_R: Tensor = None,
        optim_kwargs=None,
    ):
        dims = y_init.shape
        a = 0.5 * (t_lims[1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_cheb + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]

        C = a * f_init + y_init + a * f_init * self.t_cheb[None, 1:]

        if B_R is None:
            B_R = torch.rand(*dims, self.num_coeff_per_dim - 2, device=self.device)

        B_R.requires_grad_(True)
        optim_class = optim_kwargs["optim_class"]
        del optim_kwargs["optim_class"]
        optimizer = optim_class([B_R], **optim_kwargs)

        for i in range(self.max_iters):
            optimizer.zero_grad()

            y_k = C + B_R @ self.PHI_r
            Dy_k = B_R @ self.DPHI_r

            f_k = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                t_true[1:],
                y_k,
            )

            loss = torch.norm(Dy_k - a * (f_k - f_init))
            loss.backward()

            optimizer.step()

            if self.callback is not None:
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                self.callback(i, t_lims, y_init, f_init, B)

        return y_k[..., -1], f_k[..., -1], B_R

    def solve(
        self,
        f,
        t_span,
        y_init,
        f_init=None,
        B_init=None,
        mode="zero",
        solver_kwargs: dict = None,
    ):

        solver_iter: Callable = ...
        match mode:
            case "zero":
                solver_iter = self._zero_order
            case "gd":
                solver_iter = partial(self._gd, optim_kwargs=solver_kwargs)

        if f_init is None:
            f_init = f(t_span[0], y_init)

        y_eval = [y_init]

        Bk = None
        yk = y_init
        fk = f_init
        for t_lims in zip(t_span, t_span[1:]):
            yk, fk, Bk = self._zero_order(f, t_lims, yk, fk, Bk)
            y_eval.append(yk)

        return torch.stack(y_eval, dim=0), None
