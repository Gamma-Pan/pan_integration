import contextlib

import torch
from torch import Tensor, tensor, nn
from torch.func import vmap
from torch.autograd import Function
from typing import Tuple, Callable
import ipdb

import contextlib
from torch.profiler import profile, ProfilerActivity


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

class PanZero:
    def __init__(
            self,
            num_coeff_per_dim,
            num_points,
            delta=1e-2,
            max_iters=20,
            t_lims=None,
            device=None,
            callback=None,
    ):
        self.max_iters = max_iters
        self.callback = callback
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_points
        self.delta = delta

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # if t_lims don't change, like in a NODE training setting no need
        # to recalculate some quantities every pass
        if t_lims is not None:
            self.t_lims = t_lims

        # likewise keep previous B to pass as initial conditions
        # instead of random every time
        self.B_prev = None

    @property
    def t_lims(self):
        return self._t_lims

    @t_lims.setter
    def t_lims(self, value):
        self._t_lims = value
        (
            self.t_cheb,
            self.Phi,
            self.DPhi,
            self.Phi_tail,
            self.inv0,
            self.Phi_c,
            self.Phi_d,
        ) = self._calc_independent(
            value, self.num_coeff_per_dim, self.num_points, self.device
        )

    @staticmethod
    def _calc_independent(t_lims, num_coeff_per_dim, num_points, device):
        t_cheb = -torch.sign(t_lims[-1] - t_lims[0]) * torch.cos(
            torch.pi * (torch.arange(num_points) / num_points)
        ).to(device)

        Dt = torch.diff(torch.cat([t_cheb, tensor([1], device=device)]))

        Phi = T_grid(t_cheb, num_coeff_per_dim)
        DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t_cheb, num_coeff_per_dim)

        inv0 = torch.linalg.inv(torch.stack([Phi[0:2, 0], DPhi[0:2, 0]]).T)

        Phi_tail = torch.stack([Phi[2:, 0], DPhi[2:, 0]], dim=-1)

        Phi_b = DPhi[2:, :] - Phi_tail @ inv0 @ torch.stack(
            [DPhi[0, :], DPhi[1, :]], dim=0
        )

        # invert a (num_coeff X num_coeff) matrix once
        Q = torch.linalg.inv(Phi_b @ (Dt[:, None] * Phi_b.mT))

        Phi_b_T = Dt[:, None] * Phi_b.mT
        Phi_c = Phi_b_T @ Q

        Phi_d = inv0 @ torch.stack([DPhi[0, :], DPhi[1, :]], dim=0) @ Phi_b_T @ Q

        return t_cheb, Phi, DPhi, Phi_tail, inv0, Phi_c, Phi_d

    def _zero_order_itr(self, f, t_lims, y_init, f_init, B_init) -> Tensor:
        # if saved don't recalculate
        if self.t_lims is not None:
            t_cheb = self.t_cheb
            Phi = self.Phi
            inv0 = self.inv0
            Phi_c = self.Phi_c
            Phi_d = self.Phi_d
            Phi_tail = self.Phi_tail
        else:
            t_cheb, Phi, _, Phi_tail, inv0, Phi_c, Phi_d = self._calc_independent(
                t_lims, self.num_coeff_per_dim, self.num_points, y_init.device
            )

        yf_init = torch.stack([y_init, f_init], dim=-1)

        def add_head(B_tail):
            head = (yf_init - B_tail @ Phi_tail) @ inv0
            return torch.cat([head, B_tail], dim=-1)

        B = B_init
        for i in range(1, self.max_iters + 1):
            # if self.callback is not None:
            #     self.callback(t_lims, y_init, add_head(B))
            B_prev = B
            fapprox = vmap(f, in_dims=(0, -1), out_dims=(-1,))(
                t_cheb,
                add_head(B_prev) @ Phi,
                )

            B = fapprox @ Phi_c - yf_init @ Phi_d

            delta = torch.norm(B - B_prev)
            if delta.item() < self.delta:
                break

        return add_head(B)


    def _first_order_itr(
            self, t_span, y_init, f_init=None, B_init: Tensor | str = None
    ):
        dims = y_init.shape

    def solve(self, f, t_span, y_init, f_init=None, B_init: Tensor | str = None):
        dims = y_init.shape

        if B_init == "prev":
            B_init = self.B_prev

        if B_init is None or B_init.shape != torch.Size(
                [*dims, self.num_coeff_per_dim - 2]
        ):
            B_init = torch.rand(*dims, self.num_coeff_per_dim - 2, device=self.device)

        # costs 1 nfe
        # if f_init is None:
        f_init = f(t_span[0], y_init)

        B = self._zero_order_itr(f, (t_span[0], t_span[-1]), y_init, f_init, B_init)

        self.B_prev = torch.mean(B[..., 2:]).expand(
            *dims, self.num_coeff_per_dim - 2
        )
        t_out = -1 + 2 * (t_span - t_span[0]) / (t_span[-1] - t_span[0])
        Phi_out = T_grid(t_out, self.num_coeff_per_dim)
        approx = B @ Phi_out

        return approx.permute(-1, *list(range(0, len(dims)))), B
