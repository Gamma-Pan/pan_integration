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
            t_eval, traj = solver.solve(f, t_span, y_init)
            ctx.save_for_backward(t_eval, traj)

            return t_span, traj

        @staticmethod
        def backward(ctx, *grad_output):
            t_eval, traj = ctx.saved_tensors
            t_span = t_eval[::solver.num_points]
            y_T = traj[-1]
            batch_sz, *dims = y_T.shape
            a_y_T = grad_output[-1][-1]

            def adjoint_dynamics(t, a_y):
                # if t.dim()<1: return torch.zeros_like(a_y)

                # get corresponding trajectory interval
                # t_l = t[*(t.dim() * [-1])]
                # idx = range(len(t_span))[t_l == t_span]

                vjp_func = torch.func.vjp
                return torch.zeros_like(a_y)

            solver.solve(adjoint_dynamics, t_span.flip(0), a_y_T)

            return None
            return dL_theta, dL_y0, None

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
