import torch
from torch import Tensor, tensor, nn
from torch.autograd import Function
from torch.func import vmap
import ipdb
import lightning

from .solvers import PanSolver, T_grid


def make_pan_adjoint(f, thetas, solver: PanSolver, device=torch.device("cpu")):

    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_span):
            traj, _ = solver.solve(f, t_span, y_init)
            ctx.save_for_backward(t_span, traj)

            return t_span, traj

        @staticmethod
        def backward(ctx, *grad_output):
            t_span, traj = ctx.saved_tensors

            y_T = traj[-1]
            batch_sz, *dims = y_T.shape

            a_y_T = grad_output[-1][-1]

            f_params = torch.cat(
                [
                    v.contiguous().flatten()
                    for k, v in dict(f.named_parameters()).items()
                ]
            )
            a_theta_T = torch.zeros_like(f_params)

            y_sz, a_y_sz, a_theta_sz = y_T.numel(), a_y_T.numel(), a_theta_T.numel()
            y_shape, a_y_shape, a_theta_shape = y_T.shape, a_y_T.shape, a_theta_T.shape
            A = torch.cat([y_T.flatten(), a_y_T.flatten(), a_theta_T.flatten()])

            def func_call(t, y, theta):
                return torch.func.functional_call(f, theta, (t, y))

            def adjoint_dynamics(t, A):
                y = A[:y_sz].reshape(y_shape)
                a_y = A[y_sz : y_sz + a_y_sz].reshape(a_y_shape)

                Dy, vjp_func = torch.func.vjp(
                    func_call, t, y, dict(f.named_parameters())
                )
                _, Da_y, Da_theta_dict = vjp_func(-a_y)

                Da_theta = torch.cat(
                    [v.contiguous().flatten() for k, v in Da_theta_dict.items()]
                )
                return torch.cat([Dy.flatten(), Da_y.flatten(), Da_theta.flatten()])

            DA, _ = solver.solve(adjoint_dynamics, t_span.flip(0), A)
            grads = DA[-1]

            DL_y = grads[y_sz : y_sz + a_y_sz].reshape(a_y_shape)
            DL_theta = grads[-a_theta_sz:].reshape(a_theta_shape)
            return DL_theta, DL_y, None

    def _pan_int_adjoint(t_span, y_init):
        return _PanInt.apply(thetas, y_init, t_span)

    return _pan_int_adjoint


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        solver: PanSolver | dict,
        sensitivity="adjoint",
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

        if isinstance(solver, dict):
            solver = PanSolver(**solver, device=device)

        if sensitivity == "adjoint":
            self.pan_int = make_pan_adjoint(self.vf, self.thetas, solver, device=device)
        elif sensitivity == "autograd":
            self.pan_int = lambda t, y_init: (t, solver.solve(self.vf, t, y_init)[0])

    def forward(self, y_init, t_span, *args, **kwargs):
        return self.pan_int(t_span, y_init)
