import torch
from torch import Tensor, tensor, nn
from torch.autograd import Function
from torch.func import vmap
import ipdb
import lightning

from .solvers import PanSolver, T_grid


def make_pan_adjoint(
    f, thetas, solver: PanSolver, solver_adjoint: PanSolver, device=torch.device("cpu")
):

    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_span):
            traj = solver.solve(f, t_span, y_init)
            ctx.save_for_backward(t_span, traj)

            return t_span, traj

        @staticmethod
        def backward(ctx, *grad_output):
            # code based on torchdyn
            t_span, traj = ctx.saved_tensors
            y_T = traj[-1]
            a_y_T = grad_output[-1][-1]

            vf_params = torch.cat(
                [v.contiguous().view(-1) for k, v in dict(f.named_parameters()).items()]
            )
            a_theta_T = torch.zeros_like(vf_params)

            y_sz, a_y_sz, a_theta_sz = y_T.numel(), a_y_T.numel(), a_theta_T.numel()
            y_shape, a_y_shape, a_theta_shape = y_T.shape, a_y_T.shape, a_theta_T.shape

            A = torch.cat([y_T.flatten(), a_y_T.flatten(), a_theta_T.flatten()])

            def adjoint_dynamics(t, A):
                y, a_y, a_theta = (
                    A[:y_sz].reshape(y_shape),
                    A[y_sz : y_sz + a_y_sz].reshape(a_y_shape),
                    A[-a_theta_sz :].reshape(a_theta_shape),
                )
                dy = f(t, y)

                def func_f(theta, y):
                    return torch.func.functional_call(f, theta, (t, y))

                _, vjp_func = torch.func.vjp(func_f, dict(f.named_parameters()), y)
                *da_thetas, da_y = vjp_func(-a_y)
                # re-combine
                da_theta = torch.cat(
                    [v.flatten() for k, v in da_thetas[0].items()], dim=-1
                )

                return torch.cat([dy.flatten(), da_y.flatten(), da_theta.flatten()])

            A_traj_t0 = solver_adjoint.solve(adjoint_dynamics, t_span.flip(0), A)[-1]

            dL_y0 = A_traj_t0[y_sz : y_sz + a_y_sz].reshape(a_y_shape)
            dL_theta = A_traj_t0[-a_theta_sz:].reshape(a_theta_shape)
            return dL_theta, dL_y0, None

    def _pan_int_adjoint(t_span, y_init):
        return _PanInt.apply(thetas, y_init, t_span)

    return _pan_int_adjoint


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        solver: PanSolver | dict,
        solver_adjoint: PanSolver | dict,
        sensitivity="adjoint",
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

        if isinstance(solver, dict):
            solver = PanSolver(**solver, device=device)

        if isinstance(solver_adjoint, dict):
            solver_adjoint = PanSolver(**solver_adjoint, device=device)

        if sensitivity == "adjoint":
            self.pan_int = make_pan_adjoint(
                self.vf, self.thetas, solver, solver_adjoint, device=device
            )
        elif sensitivity == "autograd":
            self.pan_int = lambda t, y_init: (t, solver.solve(self.vf, t, y_init))

    def forward(self, y_init, t_span, *args, **kwargs):
        return self.pan_int(t_span, y_init)
