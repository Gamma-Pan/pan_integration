import contextlib

import torch
from torch import Tensor, tensor, nn
from torch.autograd import Function
import ipdb

from .solvers import PanSolver, T_grid


def make_pan_adjoint(f, thetas, solver: PanSolver, solver_adjoint: PanSolver):
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

                y_back = B_fwd @ T_n
                _, vjp_fun = torch.func.vjp(
                    lambda x: f(t, x),
                    y_back,
                )
                Da_y = -vjp_fun(a_y)[0]

                return Da_y

            t_eval_adjoint = torch.linspace(
                t_eval[-1], t_eval[0], solver_adjoint.num_points
            ).to(device)

            #########
            # import torchdyn
            # import matplotlib.pyplot as plt
            # from pan_integration.utils import plotting
            #
            # _a_y_T = torch.clone(a_y_T)
            #
            # _, test = torchdyn.numerics.odeint(
            #     adjoint_dynamics,
            #     _a_y_T,
            #     t_eval_adjoint,
            #     solver="tsit5",
            #     atol=1e-6,
            # )
            # fig = plt.gcf()
            # for ax, data in zip(fig.axes, test[:, 0, :].T):
            #     ax.plot(t_eval_adjoint, data,'g--')
            #
            # plotting.wait()
            ################

            A_traj, _ = solver_adjoint.solve(
                adjoint_dynamics, t_eval_adjoint, a_y_T, f_init=Da_y_T, B_init="prev"
            )

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

            # ipdb.set_trace()
            grads_vec = torch.cat(
                [
                    p.contiguous().flatten() if p is not None else torch.zeros(1, device=device)
                    for p in grads
                ]
            )
            DL_theta = ((t_eval[-1] - t_eval[0]) / (solver.num_points - 1)) * grads_vec

            return DL_theta, None, None

    def _pan_int_adjoint(t_span, y_init):
        return _PanInt.apply(thetas, y_init, t_span)

    return _pan_int_adjoint


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        t_span,
        solver: PanSolver | dict,
        solver_adjoint: PanSolver | dict,
        sensitivity="adjoint",
    ):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.t_span = t_span

        if isinstance(solver, dict):
            solver = PanSolver(**solver, device=t_span.device)

        if isinstance(solver_adjoint, dict):
            solver_adjoint = PanSolver(**solver_adjoint, device=t_span.device)

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
