import torch
from torch import Tensor, vstack, tensor, cat, stack, optim
from torch.func import vmap
from torch.autograd import Function

import torchdyn

from typing import Tuple, Callable

import ipdb

torch.manual_seed(42)


def T_grid(t, num_coeff_per_dim, device):
    if len(t.size()) == 0:
        t = t[None]

    num_points = len(t)

    out = torch.empty(num_points, num_coeff_per_dim, device=device)
    out[:, 0] = torch.ones(
        num_points,
    )
    out[:, 1] = t

    for i in range(2, num_coeff_per_dim):
        out[:, i] = 2 * t * out[:, i - 1] - out[:, i - 2]

    return out


def U_grid(t, num_coeff_per_dim, device):
    num_points = len(t)

    out = torch.empty(num_points, num_coeff_per_dim, device=device)
    out[:, 0] = torch.ones(
        num_points,
    )
    out[:, 1] = 2 * t

    for i in range(2, num_coeff_per_dim):
        out[:, i] = 2 * t * out[:, i - 1] - out[:, i - 2]

    return out


def DT_grid(t, num_coeff_per_dim, device):
    num_points = len(t)

    out = torch.hstack(
        [
            torch.zeros(num_points, 1, dtype=torch.float, device=device),
            torch.arange(1, num_coeff_per_dim, dtype=torch.float, device=device)
            * U_grid(t, num_coeff_per_dim - 1, device),
        ]
    )

    return out


def _euler_coarse(
    f, y_init, f_init, num_coeff_per_dim, t_lims, num_steps=5, device=None
):
    if device is None:
        device = y_init.device

    num_dims = len(y_init.shape)
    t_coarse = torch.linspace(0, 1, num_steps)

    Phi = T_grid(t_coarse, num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
    DPhi = (
        2 / (t_lims[1] - t_lims[0]) * DT_grid(t_coarse, num_coeff_per_dim, device).mT
    )  # (TxC).T = (CxT)
    inv0 = torch.linalg.inv(torch.stack([Phi[0:2, 0], DPhi[0:2, 0]]).T)

    # coarse solution, put time dimension last
    y_eul = torchdyn.numerics.odeint(f, y_init, t_coarse, solver="euler")[1].permute(
        *range(1, num_dims + 1), 0
    )

    Phi_a = torch.stack([y_init, f_init], dim=-1) @ inv0 @ Phi[0:2, :]
    Phi_b = (
        Phi[2:, :] - torch.stack([Phi[2:, 0], DPhi[2:, 0]], dim=-1) @ inv0 @ Phi[0:2, :]
    )

    Y = (y_eul - Phi_a).mT
    A = (Phi_b.mT).view(*[1 for _ in range(num_dims - 1)], *Phi_b.shape).mT
    return torch.linalg.lstsq(A, Y).solution.mT


@torch.no_grad()
def pan_int(
    f: Callable,
    t_lims: list | Tensor,
    y_init: Tensor,
    num_coeff_per_dim: int,
    num_points: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_params: dict,
    etol=1e-5,
    max_iters=20,
    coarse_iters=5,
    init: str = "euler",
    batched=True,
    callback=None,
):
    if optimizer_class is None:
        optimizer_class = optim.SGD
    if optimizer_params is None:
        optimizer_params = {"lr": 0.1}

    device = y_init.device
    if batched:
        batch_sz, *dims = y_init.shape
    else:
        batch_sz, dims = 1, y_init.shape

    f_init = f(t_lims[0], y_init)

    # t at chebyshev nodes
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    Dt = torch.diff(torch.cat([t, tensor([1])]))

    Phi = T_grid(t, num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
    DPhi = (
        2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim, device).mT
    )  # (TxC).T = (CxT)

    if init == "euler":
        B_init = _euler_coarse(
            f, y_init, f_init, num_coeff_per_dim, t_lims, device=device
        )
    elif init == "random":
        B_init = torch.rand(
            (batch_sz, *dims, num_coeff_per_dim - 2),
            device=device,
        )

    inv0 = torch.linalg.inv(torch.stack([Phi[0:2, 0], DPhi[0:2, 0]]).T)

    def head(B_tail):
        return (
            torch.stack([y_init, f_init], dim=-1)
            - B_tail @ torch.stack([Phi[2:, 0], DPhi[2:, 0]], dim=-1)
        ) @ inv0

    with torch.enable_grad():
        B = B_init
        B.requires_grad = True
        B.retain_grad()
        optimizer = optimizer_class([B], **optimizer_params)

        def loss_fn(B_tail):
            B_head = head(B_tail)
            B = torch.cat([B_head, B_tail], dim=-1)
            # consider that each element of this tensor is a scalar trajectory
            approx = B @ Phi
            Dapprox = B @ DPhi

            # vmap to avoid conflicts with convolution, normalization layers that require 3D/4D inputs
            loss = torch.sum(
                (Dapprox - vmap(f, in_dims=(0, -1), out_dims=(-1))(t, approx)) ** 2 * Dt
            )

            return loss, approx, Dapprox

        # Phi_plot =  T_grid(torch.linspace(-1,1, 200), num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
        for i in range(max_iters):
            optimizer.zero_grad()
            loss, approx, Dapprox = loss_fn(B)
            print(approx[0,:,-1])

            if callback is not None:
                callback(i,
                    approx.detach().permute(-1, *torch.arange(len(dims) + 1).tolist()),
                    Dapprox=Dapprox.detach().permute(
                        -1, *torch.arange(len(dims) + 1).tolist()
                    ),
                )

            loss.backward()
            optimizer.step()

    # return trajectory with time as first dim
    return approx.permute(-1, *torch.arange(len(dims) + 1).tolist())


def make_pan_adjoint(
    f,
    thetas,
    num_coeff_per_dim,
    num_points,
    num_coeff_per_dim_adjoint,
    num_points_adjoint,
    etol_ls=1e-5,
    max_iters_ls=20,
    callback=None,
):
    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_eval):
            traj, B_fwd = pan_int(
                f,
                t_eval,
                y_init,
                num_coeff_per_dim,
                num_points,
                etol_ls,
                max_iters_ls=max_iters_ls,
                callback=callback,
                return_B=True,
            )
            ctx.save_for_backward(t_eval, traj, B_fwd)

            return t_eval, traj

        @staticmethod
        def backward(ctx, *grad_output):
            t_eval, yS, B_fwd = ctx.saved_tensors
            device = yS.device

            points, batch_sz, dims = yS.shape

            # dL/dz(T) -> (1xBxD)
            a_y_T = grad_output[1][-1]
            with torch.set_grad_enabled(True):
                yT = yS[-1].requires_grad_(True)
                fT = f(t_eval[-1], yT)
                Da_y_T = -torch.autograd.grad(
                    fT, yT, a_y_T, allow_unused=True, retain_graph=False
                )[0]

            def adjoint_dynamics(t, a_y):
                y_back = T_grid(t, num_coeff_per_dim, device=t.device) @ B_fwd

                with torch.set_grad_enabled(True):
                    y_back.requires_grad_(True)
                    f_back = f(t_eval, y_back)

                    Da_y = -torch.autograd.grad(
                        f_back, y_back, a_y, allow_unused=True, retain_graph=False
                    )[0]

                return Da_y

            t_eval_adjoint = torch.linspace(
                t_eval[-1], t_eval[0], num_points_adjoint
            ).to(device)
            A_traj = pan_int(
                adjoint_dynamics,
                t_eval_adjoint,
                a_y_T,
                num_coeff_per_dim_adjoint,
                num_points_adjoint,
                etol_ls,
                f_init=Da_y_T,
                callback=callback,
            )

            a_y_back = A_traj.reshape(-1, dims)

            with torch.set_grad_enabled(True):
                y_back = (
                    T_grid(t_eval_adjoint, num_coeff_per_dim, device=device) @ B_fwd
                ).transpose(
                    0, 1
                )  # transpose before reshape to align with A_traj
                y_back.requires_grad_(True)
                f_back = f(t_eval_adjoint, y_back.reshape(-1, dims))

                grads = torch.autograd.grad(
                    f_back,
                    tuple(f.parameters()),
                    a_y_back,
                    allow_unused=True,
                    retain_graph=False,
                )

            grads_vec = torch.cat([p.contiguous().flatten() for p in grads])

            DL_theta = (
                torch.abs(t_eval[0] - t_eval[-1]) / (num_points_adjoint - 1)
            ) * grads_vec

            # ipdb.set_trace()

            return DL_theta, None, None

    def _pan_int_adjoint(y_init, t_eval):
        return _PanInt.apply(thetas, y_init, t_eval)

    return _pan_int_adjoint
