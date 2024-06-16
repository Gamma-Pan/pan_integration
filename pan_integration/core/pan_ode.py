import torch
from torch import Tensor, tensor, nn
from torch.autograd import Function
from torch.func import vmap
import ipdb

from .solvers import PanSolver, T_grid


def make_pan_adjoint(
    f, thetas, solver: PanSolver, solver_adjoint: PanSolver, device=torch.device("cpu")
):

    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_span):
            traj, B_all = solver.solve(f, t_span, y_init)
            ctx.save_for_backward(t_span, B_all)

            return t_span, traj

        @staticmethod
        def backward(ctx, *grad_output):
            t_fwd, Bs_fwd = ctx.saved_tensors

            a_y_T = grad_output[1][-1]

            def choose_B(t):
                # ugly way to make vmap work
                dims = Bs_fwd.shape
                slot = ((t < t_fwd[1:]) * (t >= t_fwd[:-1])).to(torch.float)
                mask = slot.reshape(-1, *(len(dims) - 1) * [1])
                B_fwd = torch.sum(Bs_fwd * mask, dim=0)
                return B_fwd

            def adjoint_dynamics(t, a_y):
                T_n = torch.cos(
                    torch.arange(solver.num_coeff_per_dim, device=t.device)
                    * torch.arccos(t)
                )

                B_fwd = choose_B(t)
                y_back = B_fwd @ T_n

                _, vjp_fun = torch.func.vjp(
                    lambda x: f(t, x),
                    y_back,
                )
                Da_y = -vjp_fun(a_y)[0]

                return Da_y

            # ###############
            # import torchdyn
            # import matplotlib.pyplot as plt
            # from ..utils.plotting import wait
            #
            # fig = plt.gcf()
            # teval, test = torchdyn.numerics.odeint(
            #     adjoint_dynamics,
            #     a_y_T,
            #     torch.linspace(1, 0, 100),
            #     solver="tsit5",
            #     atol=1e-4,
            # )
            # for idx, ax in enumerate(fig.axes):
            #     ax.plot(
            #         torch.linspace(1, 0, 100), test[:, 0, idx].detach().numpy(), "b--"
            #     )
            #     ax.set_ylim(torch.min(test) * 1.1, torch.max(test) * 1.1)
            #     ax.autoscale(enable=False)
            #
            # wait()
            # ################

            traj_pan, Bs_back = solver_adjoint.solve(
                adjoint_dynamics, t_fwd.flip(0), a_y_T
            )
            # get N/intervals points for every interval
            num_inters = Bs_back.shape[0]
            N = (100 // num_inters) * num_inters

            Phi = (
                T := T_grid(
                    torch.linspace(
                        -1, 1 - (2 / N), N // num_inters, device=a_y_T.device
                    ),
                    solver.num_coeff_per_dim,
                )
            ).expand(num_inters, *T.shape)

            y_back = vmap(torch.matmul, in_dims=(0, 0), out_dims=(0))(Bs_fwd, Phi)

            Phi = (
                T := T_grid(
                    torch.linspace(-1, 1 - (2 / N), N // num_inters, device=a_y_T.device),
                    solver_adjoint.num_coeff_per_dim,
                )
            ).expand(num_inters, *T.shape)

            a_y_back = vmap(torch.matmul, in_dims=(0, 0), out_dims=(0))(Bs_back, Phi)

            y_back = y_back.permute(-1, *range(len(y_back.shape) - 1)).reshape(
                -1, *a_y_T.shape[1:]
            )
            a_y_back = a_y_back.permute(-1, *range(len(a_y_back.shape) - 1)).reshape(
                -1, *a_y_T.shape[1:]
            )

            # all good up to here
            with torch.set_grad_enabled(True):

                y_back.requires_grad_(True)
                f_back = f(torch.linspace(t_fwd[0], t_fwd[-1], N), y_back)

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
                    (
                        p.contiguous().flatten()
                        if p is not None
                        else torch.zeros(1, device=device)
                    )
                    for p in grads
                ]
            )
            DL_theta = ((t_fwd[-1] - t_fwd[0]) / (N - 1)) * grads_vec

            return DL_theta, None, None

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
            self.pan_int = lambda t, y_init: (t, solver.solve(self.vf, t, y_init)[0])

    def forward(self, y_init, t_span, *args, **kwargs):
        return self.pan_int(t_span, y_init)
