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


class PanSolver(nn.Module):
    def __init__(
        self,
        num_coeff_per_dim,
        num_points,
        deltas=(1e-2, 1e-5),
        max_iters=(20, 10),
        optim=None,
        t_lims=None,
        device=None,
        callback=None,
    ):
        super().__init__()

        self.optim = optim
        self.callback = callback
        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_points

        self.delta_zero = deltas[0]
        self.delta_one = deltas[1]

        self.max_iters_zero = max_iters[0]
        self.max_iters_one = max_iters[1]

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self._t_lims = t_lims
        self.B_prev = None

    @property
    def t_lims(self):
        return self._t_lims

    @t_lims.setter
    def t_lims(self, value):
        # if t_lims don't change, like in a NODE training setting no need
        # to recalculate some quantities every pass
        if value is not None:
            self._t_lims = value
            (
                self.t_cheb,
                self.t_true,
                self.Dt,
                self.Phi,
                self.DPhi,
                self.Phi_tail,
                self.inv0,
                self.Phi_c,
                self.Phi_d,
            ) = self._calc_independent(
                value,
                self.num_coeff_per_dim,
                self.num_points,
                self.device,
            )

    @staticmethod
    def _calc_independent(t_lims, num_coeff_per_dim, num_points, device):
        t_cheb = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(
            device
        )
        t_true = t_lims[0] + 0.5 * (t_lims[-1] - t_lims[0]) * (t_cheb + 1)

        Dt = torch.diff(torch.cat([t_cheb, tensor([1], device=device)]))

        Phi = T_grid(t_cheb, num_coeff_per_dim)
        DPhi = 2 / (t_lims[-1] - t_lims[0]) * DT_grid(t_cheb, num_coeff_per_dim)

        inv0 = torch.linalg.inv(torch.stack([Phi[0:2, 0], DPhi[0:2, 0]]).T)

        Phi_tail = torch.stack([Phi[2:, 0], DPhi[2:, 0]], dim=-1)

        Phi_b = DPhi[2:, :] - Phi_tail @ inv0 @ torch.stack(
            [DPhi[0, :], DPhi[1, :]], dim=0
        )

        # invert a (num_coeff X num_coeff) matrix once
        # Q = torch.linalg.inv(Phi_b @ (Dt[:, None] * Phi_b.mT))
        Q = Phi_b @ (Dt[:, None] * Phi_b.mT)

        Phi_b_T = Dt[:, None] * Phi_b.mT
        # Phi_c = Phi_b_T @ Q
        Phi_c = torch.linalg.solve(Q, Phi_b_T, left=False)

        # Phi_d = inv0 @ torch.stack([DPhi[0, :], DPhi[1, :]], dim=0) @ Phi_b_T @ Q
        Phi_d = torch.linalg.solve(
            Q, inv0 @ torch.stack([DPhi[0, :], DPhi[1, :]], dim=0) @ Phi_b_T, left=False
        )

        return t_cheb, t_true, Dt, Phi, DPhi, Phi_tail, inv0, Phi_c, Phi_d

    def _solver_itr(self, f, t_lims, y_init, f_init, B_init) -> Tensor:
        # if saved don't recalculate
        if self.t_lims is not None:
            t_cheb = self.t_cheb
            t_true = self.t_true
            Dt = self.Dt
            Phi = self.Phi
            DPhi = self.DPhi
            inv0 = self.inv0
            Phi_c = self.Phi_c
            Phi_d = self.Phi_d
            Phi_tail = self.Phi_tail
        else:
            (
                t_cheb,
                t_true,
                Dt,
                Phi,
                DPhi,
                Phi_tail,
                inv0,
                Phi_c,
                Phi_d,
            ) = self._calc_independent(
                t_lims, self.num_coeff_per_dim, self.num_points, y_init.device
            )

        batch_sz, *dims = y_init.shape
        yf_init = torch.stack([y_init, f_init], dim=-1)

        def add_head(B_tail):
            head = (yf_init - (B_tail @ Phi_tail)) @ inv0
            return torch.cat([head, B_tail], dim=-1)

        B_f = B_init
        ## zero order
        B = B_init
        for i in range(1, self.max_iters_zero + 1):
            if self.callback is not None:
                self.callback(t_lims, y_init, add_head(B_f))

            B_prev = B
            fapprox = vmap(f, in_dims=(0, -1), out_dims=(-1,))(
                t_true, (add_head(B_prev) @ Phi)
            )

            B = (fapprox @ Phi_c) - yf_init @ Phi_d

            N = 20
            if i < N:
                B_f = B
            else:
                B_f = (1 / i) * (B_f * (i - 1) + B)
            # if i % 10 == 0:
            #     delta = torch.norm(B - B_prev)
            #     if delta.item() < self.delta_zero:
            #         break

        return add_head(B_f)

    # def _gd_itr(self):
    #     B.requires_grad = True
    #     optimizer = self.optim["optimizer_class"]([B], **self.optim["params"])
    #
    #     def loss_fn(B_tail):
    #         Bl = add_head(B_tail)
    #         Dapprox = Bl @ DPhi
    #         fapprox = vmap(f, in_dims=(0, -1), out_dims=(-1,))(t_true, Bl @ Phi)
    #         loss = torch.sum((Dapprox - fapprox) ** 2 * Dt)
    #
    #         return loss
    #
    #     breakflag = False
    #
    #     def closure():
    #         optimizer.zero_grad()
    #         with torch.enable_grad():
    #             loss = loss_fn(B)
    #             loss.backward(retain_graph=True)
    #
    #         return loss
    #
    #     for i in range(1, self.max_iters_one + 1):
    #         optimizer.step(closure)
    #
    #         if self.callback is not None:
    #             self.callback(t_lims, y_init.cpu(), add_head(B).clone().detach().cpu())
    #
    #     return add_head(B).detach()

    def solve(self, f, t_span, y_init, f_init=None, B_init: Tensor | str = None):
        dims = y_init.shape

        if B_init == "prev":
            B_init = self.B_prev

        if B_init is None or B_init.shape != torch.Size(
            [*dims, self.num_coeff_per_dim - 2]
        ):
            B_init = torch.rand(*dims, self.num_coeff_per_dim - 2, device=self.device)

        # costs 1 nfe
        if f_init is None:
            f_init = f(t_span[0], y_init)

        B = self._solver_itr(f, (t_span[0], t_span[-1]), y_init, f_init, B_init)

        self.B_prev = torch.mean(B[..., 2:]).expand(*dims, self.num_coeff_per_dim - 2)
        t_out = -1 + 2 * (t_span - t_span[0]) / (t_span[-1] - t_span[0])
        Phi_out = T_grid(t_out, self.num_coeff_per_dim)
        approx = B @ Phi_out

        return approx.permute(-1, *range(0, len(dims))), B


class PanSolver2(nn.Module):
    def __init__(self, num_coeff_per_dim, max_iters, delta, callback=None, device=None):
        super().__init__()
        self.num_coeff_per_dim = num_coeff_per_dim
        self.inner_points = 0
        self.B_prev = None
        self.callback = callback
        self.max_iters = max_iters
        self.delta = delta

    def _solver_itr(self, f, t_span, y_init, f_init, B_init):
        # f_init for compatability
        dims = y_init.shape
        device = y_init.device

        t_lims = [t_span[0], t_span[-1]]

        B = B_init

        t_cheb = -torch.cos(
            torch.pi
            * (torch.arange(self.num_coeff_per_dim) / (self.num_coeff_per_dim - 1))
        ).to(device)

        t_simpsons = torch.empty(2 * len(t_cheb) - 1, device=device)
        t_simpsons[1::2] = 0.5 * (t_cheb[1:] + t_cheb[:-1])
        t_simpsons[::2] = t_cheb

        t_true = (t_lims[0] + 0.5 * (t_lims[-1] - t_lims[0]) * (t_simpsons + 1)).to(
            device
        )

        Phi_f = T_grid(t_simpsons, self.num_coeff_per_dim)
        Phi = Phi_f[:, 0::2]

        S = torch.empty((*dims, self.num_coeff_per_dim), device=device)
        S[..., 0] = 0

        for i in range(self.max_iters):
            approx = B @ Phi_f
            fapprox = vmap(f, in_dims=(0, -1), out_dims=(-1,))(t_true, approx)

            # calculate integrals using simpsons rule
            for i in range(0, self.num_coeff_per_dim - 1):
                a = t_true[2 * i]
                b = t_true[2 * i + 2]
                s = ((b - a) / 6) * (
                    fapprox[..., 2 * i]
                    + 4 * fapprox[..., 2 * i + 1]
                    + fapprox[..., 2 * i + 2]
                )

                S[..., i + 1] = S[..., i] + s

            B = torch.linalg.solve(Phi, S + y_init[..., None], left=False)

            if self.callback:
                self.callback(t_lims, y_init, B.detach().cpu())

        return B

    def solve(self, f, t_span, y_init, B_init):

        dims = y_init.shape

        B_init = torch.rand(*dims, self.num_coeff_per_dim, device=y_init.device)

        B = self._solver_itr(f, (t_span[0], t_span[-1]), y_init, None, B_init)

        t_out = -1 + 2 * (t_span - t_span[0]) / (t_span[-1] - t_span[0])
        Phi_out = T_grid(t_out, self.num_coeff_per_dim)
        approx = B @ Phi_out

        return approx.permute(-1, *range(0, len(dims))), B
