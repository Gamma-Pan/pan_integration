import torch
from torch import Tensor, vstack, tensor, cat, stack, optim
from torch.func import vmap
from torch.autograd import Function

import torchdyn

from typing import Tuple, Callable

import ipdb

# torch.manual_seed(42)


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
    f,
    y_init,
    f_init,
    num_coeff_per_dim,
    t_lims,
    num_steps=5,
    device=None,
):
    if device is None:
        device = y_init.device

    num_dims = len(y_init.shape)
    t_coarse = -torch.cos(torch.pi * (torch.arange(num_steps) / num_steps)).to(device)

    Phi = T_grid(t_coarse, num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
    DPhi = (
        2 / (t_lims[1] - t_lims[0]) * DT_grid(t_coarse, num_coeff_per_dim, device).mT
    )  # (TxC).T = (CxT)
    inv0 = torch.linalg.inv(torch.stack([Phi[0:2, 0], DPhi[0:2, 0]]).T)

    # coarse solution, put time dimension last
    y_eul = torchdyn.numerics.odeint(f, y_init, t_coarse, solver="euler")[1]

    Phi_a = torch.stack([y_init, f_init], dim=-1) @ inv0 @ Phi[0:2, :]
    Phi_b = (
        Phi[2:, :] - torch.stack([Phi[2:, 0], DPhi[2:, 0]], dim=-1) @ inv0 @ Phi[0:2, :]
    )

    Y = (y_eul.permute(*range(1, num_dims + 1), 0) - Phi_a).mT
    A = Phi_b.view(*[1 for _ in range(num_dims - 1)], *Phi_b.shape).mT
    return torch.linalg.lstsq(A, Y).solution.mT


@torch.no_grad()
def zero_order_int(
    f: Callable,
    t_lims: list | Tensor,
    y_init: Tensor,
    num_coeff_per_dim: int,
    num_points: int,
    delta=1e-5,
    max_iters=20,
    B_init=None,
    callback=None,
    f_init=None,
    Phi=None,
    DPhi=None,
):
    device = y_init.device
    dims = y_init.shape

    if f_init is None:
        f_init = f(t_lims[0], y_init)

    # t at chebyshev nodes
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    Dt = torch.diff(torch.cat([t, tensor([1], device=device)]))

    if Phi is None:
        Phi = T_grid(t, num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
    if DPhi is None:
        DPhi = (
            2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim, device).mT
        )  # (TxC).T = (CxT)

    if callback is not None:
        t_plot = torch.linspace(-1, 1, 100)
        Phi_plot = T_grid(t_plot, num_coeff_per_dim, device).mT

        DPhi_plot = (
            2 / (t_lims[1] - t_lims[0]) * DT_grid(t_plot, num_coeff_per_dim, device).mT
        )  # (TxC).T = (CxT)

    inv0 = torch.linalg.inv(torch.stack([Phi[0:2, 0], DPhi[0:2, 0]]).T)

    def head(B_tail):
        return (
            torch.stack([y_init, f_init], dim=-1)
            - B_tail @ torch.stack([Phi[2:, 0], DPhi[2:, 0]], dim=-1)
        ) @ inv0

    Phi_a = (
        torch.stack([y_init, f_init], dim=-1)
        @ inv0
        @ torch.stack([DPhi[0, :], DPhi[1, :]], dim=0)
    )
    Phi_b = DPhi[2:, :] - torch.stack(
        [Phi[2:, 0], DPhi[2:, 0]], dim=-1
    ) @ inv0 @ torch.stack([DPhi[0, :], DPhi[1, :]], dim=0)

    # invert a (num_coeff X num_coeff) matrix once
    Q = torch.linalg.inv(Phi_b @ (Dt[:, None] * Phi_b.mT))

    B = B_init
    i = 0
    for i in range(max_iters):
        if callback is not None:
            B_plot = torch.cat([head(B), B], dim=-1)
            fapprox = B_plot @ Phi_plot
            Dapprox = B_plot @ DPhi_plot

            callback(
                t_lims[0],
                fapprox.permute(-1, *torch.arange(len(dims)).tolist()),
                Dapprox=Dapprox.permute(-1, *torch.arange(len(dims)).tolist()),
            )

        B_prev = B
        fapprox = vmap(f, in_dims=(0, -1), out_dims=(-1,))(
            t, torch.cat([head(B_prev), B_prev], dim=-1) @ Phi
        )
        B = (fapprox @ (Dt[:, None] * Phi_b.mT) - Phi_a @ (Dt[:, None] * Phi_b.mT)) @ Q

        tol = torch.norm(B - B_prev)
        if tol < delta:
            break

        B_out = torch.cat([head(B), B], dim=-1)
        solver_loss = torch.sum((B_out @ Phi - fapprox) ** 2 * Dt)

    return torch.cat([head(B), B], dim=-1), (solver_loss, i)


@torch.no_grad()
def first_order_int(
    f: Callable,
    t_lims: list | Tensor,
    y_init: Tensor,
    num_coeff_per_dim: int,
    num_points: int,
    optimizer_class: torch.optim.Optimizer = None,
    optimizer_params: dict = None,
    etol=1e-5,
    max_iters=20,
    B_init=None,
    callback=None,
    f_init=None,
    Phi=None,
    DPhi=None,
):
    if optimizer_class is None:
        optimizer_class = torch.optim.Adam
    if optimizer_params is None:
        optimizer_params = dict(lr=1e-3)

    device = y_init.device
    dims = y_init.shape

    if f_init is None:
        f_init = f(t_lims[0], y_init)

    # t at chebyshev nodes
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    Dt = torch.diff(torch.cat([t, tensor([1], device=device)]))

    if Phi is None:
        Phi = T_grid(t, num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
    if DPhi is None:
        DPhi = (
            2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim, device).mT
        )  # (TxC).T = (CxT)

    if callback is not None:
        t_plot = torch.linspace(-1, 1, 100)
        Phi_plot = T_grid(t_plot, num_coeff_per_dim, device).mT

        DPhi_plot = (
            2 / (t_lims[1] - t_lims[0]) * DT_grid(t_plot, num_coeff_per_dim, device).mT
        )  # (TxC).T = (CxT)

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
        grad_norm = torch.inf

        def loss_fn(B_tail):
            B_head = head(B_tail)
            B = torch.cat([B_head, B_tail], dim=-1)
            # consider that each element of this tensor is a scalar trajectory
            approx = B @ Phi
            Dapprox = B @ DPhi

            # vmap to avoid conflicts with convolutions, normalization layers, ... that require 3D/4D inputs
            loss = torch.sum(
                (Dapprox - vmap(f, in_dims=(0, -1), out_dims=(-1))(t, approx)) ** 2 * Dt
            )

            return loss, approx, Dapprox

        def closure():
            optimizer.zero_grad()
            loss, approx, Dapprox = loss_fn(B)

            if callback is not None:
                B_plot = torch.cat([head(B.detach()), B.detach()], dim=-1)
                approx_plot = (B_plot @ Phi_plot).permute(
                    -1, *torch.arange(len(dims)).tolist()
                )
                Dapprox_plot = (B_plot @ DPhi_plot).permute(
                    -1, *torch.arange(len(dims)).tolist()
                )
                callback(t_lims[0], approx_plot, Dapprox=Dapprox_plot)

            loss.backward()
            nonlocal grad_norm
            grad_norm = torch.norm(B.grad)
            return loss

        for i in range(max_iters):
            optimizer.step(closure)

            if grad_norm < etol:
                break

    return torch.cat([head(B), B], dim=-1)


def pan_int(
    f: Callable,
    t_span: list | Tensor,
    y_init: Tensor,
    num_coeff_per_dim: int,
    num_points: int,
    tol_zero=1e-3,
    tol_one=1e-5,
    max_iters_zero=10,
    max_iters_one=10,
    optimizer_class=None,
    optimizer_params=None,
    init="random",
    coarse_steps=5,
    f_init=None,
    callback=None,
    metrics=False,
):
    device = y_init.device
    dims = y_init.shape
    t_lims = [t_span[0], t_span[-1]]

    if f_init is None:
        f_init = f(t_lims[0], y_init)

    # t at chebyshev nodes
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)

    Phi = T_grid(t, num_coeff_per_dim, device).mT  # (TxC).T = (CxT)
    DPhi = (
        2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim, device).mT
    )  # (TxC).T = (CxT)

    if init == "random":
        B_init = torch.rand(
            (*dims, num_coeff_per_dim - 2),
            device=device,
        )
    elif init == "euler":
        B_init = _euler_coarse(
            f,
            y_init,
            f_init,
            num_coeff_per_dim,
            t_lims,
            device=device,
            num_steps=coarse_steps,
        )
    else:
        raise Exception("Invalid init type")

    B_out, (solver_loss, iters_zero) = zero_order_int(
        f,
        t_lims,
        y_init,
        num_coeff_per_dim,
        num_points,
        tol_zero,
        max_iters_zero,
        B_init,
        callback,
        f_init,
        Phi,
        DPhi,
    )

    # B_out = first_order_int(
    #     f,
    #     t_lims,
    #     y_init,
    #     num_coeff_per_dim,
    #     num_points,
    #     optimizer_class,
    #     optimizer_params,
    #     tol_one,
    #     max_iters_one,
    #     B_zero[..., 2:],
    #     callback,
    #     f_init,
    #     Phi,
    #     DPhi,
    # )

    t_out = -1 + 2 * (t_span - t_span[0]) / (t_span[-1] - t_span[0])
    Phi_out = T_grid(t_out, num_coeff_per_dim, device).mT
    approx = B_out @ Phi_out
    # put time dimension in front
    if metrics:
        return (
            approx.permute(-1, *torch.arange(len(dims)).tolist()),
            B_out,
            (solver_loss, iters_zero),
        )
    else:
        return approx.permute(-1, *torch.arange(len(dims)).tolist()), B_out


def make_pan_adjoint(
    f,
    thetas,
    num_coeff_per_dim: int,
    num_points: int,
    tol_zero=1e-3,
    tol_one=1e-5,
    max_iters_zero=10,
    max_iters_one=10,
    optimizer_class=None,
    optimizer_params=None,
    init="random",
    coarse_steps=5,
    callback=None,
    metrics=False,
):
    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_eval):
            traj, B_fwd, zero_metrics = pan_int(
                f,
                t_eval,
                y_init,
                num_coeff_per_dim,
                num_points,
                tol_zero,
                tol_one,
                max_iters_zero,
                max_iters_one,
                optimizer_class,
                optimizer_params,
                init,
                coarse_steps,
                None,
                callback,
                metrics,
            )
            ctx.save_for_backward(t_eval, traj, B_fwd)

            return t_eval, traj, zero_metrics

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

            a_theta_sz = torch.numel(thetas)
            a_theta_T = torch.zeros(a_theta_sz, device=device)
            a_t_T = vmap(lambda x, y: torch.sum(x * y))(a_y_T, fT.detach()).unsqueeze(
                -1
            )

            def adjoint_dynamics(t, a_y):
                T_n = torch.cos(
                    torch.arange(num_coeff_per_dim, device=t.device) * torch.arccos(t)
                )
                y_back = B_fwd @ T_n

                _, vjp_fun = torch.func.vjp(
                    lambda x: f(t, x),
                    y_back,
                )
                Da_y = vjp_fun(a_y)[0]

                return Da_y

            t_eval_adjoint = torch.linspace(t_eval[-1], t_eval[0], num_points).to(
                device
            )

            A_traj, _ = pan_int(
                adjoint_dynamics,
                t_eval_adjoint,
                a_y_T,
                num_coeff_per_dim,
                num_points,
                tol_zero=tol_zero,
                max_iters_zero=max_iters_zero,
                f_init=Da_y_T,
                callback=callback,
                init=init,
                coarse_steps=coarse_steps,
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
                            num_coeff_per_dim,
                            device=device,
                        ).mT
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

            DL_theta = ((t_eval[-1] - t_eval[0]) / (num_points - 1)) * grads_vec

            # ipdb.set_trace()

            return DL_theta, None, None

    def _pan_int_adjoint(y_init, t_eval):
        return _PanInt.apply(thetas, y_init, t_eval)

    return _pan_int_adjoint
