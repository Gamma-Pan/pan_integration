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

        self.counter = 0

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

            fapprox = vmap(f, in_dims=(0, -1), out_dims=(-1,))(
                t_cheb,
                add_head(B) @ Phi,
            )

            B = fapprox @ Phi_c - yf_init @ Phi_d

            # delta = torch.norm(B - B_prev)
            # if delta.to(torch.device('cpu')).item() < self.delta:
            #     break

        return add_head(B)

    def _first_order_itr(
        self, t_span, y_init, f_init=None, B_init: Tensor | str = None
    ):
        dims = y_init.shape

    def solve(self, f, t_span, y_init, f_init=None, B_init: Tensor | str = None):
        dims = y_init.shape

        # if B_init == "prev":
        #     B_init = self.B_prev
        #
        # if B_init is None or B_init.shape != torch.Size(
        #     [*dims, self.num_coeff_per_dim - 2]
        # ):
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


        self.counter += 1

        return approx.permute(-1, *list(range(0, len(dims)))), B


def make_pan_adjoint(f, thetas, solver: PanZero, solver_adjoint: PanZero):
    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_span):
            traj, B = solver.solve(f, t_span, y_init, B_init="prev")
            ctx.save_for_backward(t_span, traj, B)

            return t_span, traj

        @staticmethod
        def backward(ctx, *grad_output):
            t_eval, y_fwd, B_fwd = ctx.saved_tensors
            device = y_fwd.device

            points, batch_sz, *dims = y_fwd.shape

            # dL/dz(T) -> (1xBxD)
            a_y_T = grad_output[1][-1]
            with torch.set_grad_enabled(True):
                yT = y_fwd[-1].requires_grad_(True)
                fT = f(t_eval[-1], yT)
                Da_y_T = -torch.autograd.grad(
                    fT, yT, a_y_T, allow_unused=True, retain_graph=False
                )[0]

            # a_theta_sz = torch.numel(thetas)
            # a_theta_T = torch.zeros(a_theta_sz, device=device)
            # a_t_T = vmap(lambda x, y: torch.sum(x * y))(a_y_T, fT.detach()).unsqueeze(
            #     -1
            # )

            def adjoint_dynamics(t, a_y):
                T_n = torch.cos(
                    torch.arange(solver.num_coeff_per_dim, device=t.device)
                    * torch.arccos(t)
                )

                # ipdb.set_trace()
                y_back = B_fwd @ T_n

                _, vjp_fun = torch.func.vjp(
                    lambda x: f(t, x),
                    y_back,
                )
                Da_y = vjp_fun(a_y)[0]

                return Da_y

            t_eval_adjoint = torch.linspace(
                t_eval[-1], t_eval[0], solver_adjoint.num_points
            ).to(device)

            A_traj, _ = solver_adjoint.solve(
                adjoint_dynamics, t_eval_adjoint, a_y_T, f_init=Da_y_T, B_init="prev"
            )

            # import torchdyn
            # _, test = torchdyn.numerics.odeint(
            #     adjoint_dynamics,
            #     a_y_T,
            #     t_eval_adjoint,
            #     solver="tsit5",
            # )
            # print(torch.norm( test[-1] - A_traj[-1]))

            a_y_back = A_traj.reshape(-1, *dims)

            with torch.set_grad_enabled(True):
                y_back = (
                    (
                        B_fwd
                        @ T_grid(
                            -1
                            + 2
                            * (t_eval_adjoint - t_eval[-1])
                            / (t_eval[0] - t_eval[-1]),
                            solver.num_coeff_per_dim,
                        )
                    )
                    .permute(-1, *torch.arange(len(dims) + 1).tolist())
                    .reshape(-1, *dims)
                )

                y_back.requires_grad_(True)
                f_back = f(t_eval_adjoint, y_back)

                grads = torch.autograd.grad(
                    f_back,
                    tuple(f.parameters()),
                    a_y_back,
                    allow_unused=True,
                    retain_graph=False,
                )

            grads_vec = torch.cat([p.contiguous().flatten() for p in grads])
            DL_theta = ((t_eval[-1] - t_eval[0]) / (solver.num_points - 1)) * grads_vec
            # ipdb.set_trace()
            f.nfe = 0

            return DL_theta, None, None

    def _pan_int_adjoint(t_span, y_init):
        return _PanInt.apply(thetas, y_init, t_span)

    return _pan_int_adjoint


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        t_span,
        solver: PanZero | dict,
        solver_adjoint: PanZero | dict,
        sensitivity="adjoint",
    ):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.t_span = t_span

        if isinstance(solver, dict):
            solver = PanZero(**solver, device=t_span.device)

        if isinstance(solver_adjoint, dict):
            solver_adjoint = PanZero(**solver_adjoint, device=t_span.device)

        solver.t_lims = [t_span[0], t_span[-1]]
        solver_adjoint.t_lims = [t_span[-1], t_span[0]]

        if sensitivity == "adjoint":
            self.pan_int = make_pan_adjoint(
                self.vf,
                self.thetas,
                solver,
                solver_adjoint,
            )
        elif sensitivity == "autograd":
            self.pan_int = lambda t, y_init: (
                t,
                solver.solve(self.vf, t, y_init, B_init="prev")[0],
            )

    def forward(self, y_init, t_span, *args, **kwargs):
        _, traj = self.pan_int(t_span, y_init)
        return self.t_span, traj
