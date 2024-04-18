import torch
from torch import Tensor, vstack, tensor, cat, stack
from torch.linalg import inv
from torch.func import vmap
from ..optim import newton
from typing import Tuple
from functools import partial

from torch.autograd import Function

import pdb

# torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


def T_grid(t, num_coeff_per_dim):
    num_points = len(t)

    out = torch.empty(num_points, num_coeff_per_dim)
    out[:, 0] = torch.ones(
        num_points,
    )
    out[:, 1] = t

    for i in range(2, num_coeff_per_dim):
        out[:, i] = 2 * t * out[:, i - 1] - out[:, i - 2]

    return out


def U_grid(t, num_coeff_per_dim):
    num_points = len(t)

    out = torch.empty(
        num_points,
        num_coeff_per_dim,
    )
    out[:, 0] = torch.ones(
        num_points,
    )
    out[:, 1] = 2 * t

    for i in range(2, num_coeff_per_dim):
        out[:, i] = 2 * t * out[:, i - 1] - out[:, i - 2]

    return out


def DT_grid(t, num_coeff_per_dim):
    num_points = len(t)

    out = torch.hstack(
        [
            torch.zeros(num_points, 1, dtype=torch.float),
            torch.arange(1, num_coeff_per_dim, dtype=torch.float)
            * U_grid(t, num_coeff_per_dim - 1),
        ]
    )

    return out


def _B_init_cond(B_tail, y_init, f_init, Phi, DPhi):
    inv0 = inv(stack((Phi[:, 0, [0, 1]], DPhi[:, 0, [0, 1]]), dim=1))

    l = lambda B: inv0 @ (
        stack((y_init, f_init), dim=1)
        - cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1) @ B_tail
    )

    return torch.cat([l(B_tail), B_tail], dim=1)


# @torch.no_grad()
# def _coarse_euler_init(f, y_init, num_solver_steps, t_lims, num_coeff_per_dim):
#     dims = y_init.shape[0]
#     step = t_lims[1] - t_lims[0]
#     step_size = step / num_solver_steps
#     device = y_init.device
#
#     y_eul = torch.zeros(num_solver_steps, dims).to(device)
#     y_eul[0, :] = y_init
#     f_eul = torch.zeros(num_solver_steps, dims).to(device)
#     f_eul[0, :] = f(y_init)
#
#     # forward euler
#     for i in range(1, num_solver_steps):
#         y_cur = y_eul[i - 1, :]
#         f_cur = f(y_cur)
#         f_eul[i, :] = f_cur
#         y_eul[i, :] = y_cur + step_size * f_cur
#
#     Phi, DPhi = _cheb_phis(num_solver_steps, num_coeff_per_dim, t_lims)
#
#     inv0 = inv(vstack((Phi[0, [0, 1]], DPhi[0, [0, 1]])))
#     PHI = -vstack([Phi[:, [0, 1]], DPhi[:, [0, 1]]]) @ inv0 @ vstack(
#         [Phi[[0], 2:], DPhi[[0], 2:]]
#     ) + vstack([Phi[:, 2:], DPhi[:, 2:]])
#
#     Y = vstack([y_eul, f_eul]) - vstack(
#         [Phi[:, [0, 1]], DPhi[:, [0, 1]]]
#     ) @ inv0 @ vstack([y_init, f_eul[[0], :]])
#
#     B_ls = torch.linalg.lstsq(PHI, Y, driver="gels").solution
#     return B_ls.T.reshape(-1)


def lst_sq_solver(
    f,
    t_lims,
    y_init,
    num_coeff_per_dim,
    num_points,
    f_init=None,
    Phi=None,
    DPhi=None,
    B_init=None,
    max_steps=50,
    etol=1e-5,
    callback=None,
) -> Tensor | Tuple[Tensor, int]:
    device = y_init.device
    torch.set_default_dtype(torch.float64)
    B_init = B_init.to(torch.float64)

    t_hat = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    t = (t_lims[0] + (t_hat - t_hat[0]) *
         (t_lims[1] - t_lims[0]) / (t_hat[-1] - t_hat[0]
    ))

    batches, dims = y_init.shape

    if f_init is None:
        f_init = f(t[0], y_init)

    f_init = f_init.to(torch.float64)
    y_init = y_init.to(torch.float64)

    if Phi is None or DPhi is None:
        Phi = T_grid(t_hat, num_coeff_per_dim)[None]
        DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t_hat, num_coeff_per_dim)[None]

    # approximating Rieman integral with non uniformly spaced points requires to multiply
    # with the interval lenghts between points
    # TODO: approximate integral with sum using trapezoids instead of rectangles
    d = torch.diff(torch.cat((t, tensor(t_lims[1])[None])))[:, None].to(device)

    inv0 = inv(stack((Phi[:, 0, [0, 1]], DPhi[:, 0, [0, 1]]), dim=1))
    Phi_aT = DPhi[:, :, [0, 1]] @ inv0 @ stack((y_init, f_init), dim=1).to(device)
    Phi_bT = (
        -DPhi[:, :, [0, 1]] @ inv0 @ cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1)
        + DPhi[:, :, 2:]
    ).to(device)
    l = lambda B: inv0 @ (
        stack((y_init, f_init), dim=1)
        - cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1) @ B
    )

    Q = inv(Phi_bT.mT @ (d * Phi_bT))
    # MAIN LOOP
    B = B_init
    for i in range(max_steps):
        if callback is not None:
            callback(cat((l(B), B), dim=1))

        B_prev = B
        # B update
        B = Q @ (
            Phi_bT.mT
            @ (
                d
                * f(
                    t.to(torch.float32), (Phi @ cat((l(B), B), dim=1)).to(torch.float32)
                ).to(torch.float64)
            )
            - Phi_bT.mT @ (d * Phi_aT)
        )

        if torch.norm(B - B_prev) < etol:
            break

    torch.set_default_dtype(torch.float32)
    return cat((l(B), B), dim=1).to(torch.float32)


def newton_solver(
    f,
    t_lims,
    y_init,
    num_coeff_per_dim,
    num_points,
    f_init=None,
    Phi=None,
    DPhi=None,
    B_init=None,
    max_steps=50,
    etol=1e-5,
) -> Tensor:
    device = y_init.device
    torch.set_default_dtype(torch.float64)
    B_init = B_init.to(torch.float64)

    t_hat = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    t = t_lims[0] + (t_hat - t_hat[0]) * (
        (t_lims[1] - t_lims[0]) / (t_hat[-1] - t_hat[0])
    )
    batches, dims = y_init.shape

    if f_init is None:
        f_init = f(t[0], y_init)

    f_init = f_init.to(torch.float64)
    y_init = y_init.to(torch.float64)

    if Phi is None or DPhi is None:
        Phi = T_grid(t_hat, num_coeff_per_dim).to(device)
        DPhi = (
            2 / (t_lims[1] - t_lims[0]) * DT_grid(t_hat, num_coeff_per_dim).to(device)
        )

    # approximating Rieman integral with non uniformly spaced points requires to multiply
    # with the interval lenghts between points
    # TODO: approximate integral with sum using trapezoids instead of rectangles
    d = torch.diff(torch.cat((t, tensor(t_lims[1])[None])))[:, None].to(device)

    inv0 = inv(cat((Phi[0, [0, 1]], DPhi[0, [0, 1]]), dim=0))
    l = lambda B, y_init, f_init: inv0 @ (
        stack((y_init, f_init), dim=0) - cat((Phi[0:1, 2:], DPhi[0:1, 2:]), dim=0) @ B
    )

    # define error for one sample of batch as a function of B vectorised
    def error(B_vec, y_init, f_init):
        B_tail = B_vec.reshape(num_coeff_per_dim - 2, dims)
        B_head = l(B_tail, y_init, f_init)

        B = torch.cat((B_head, B_tail), dim=0)

        error = (
            (1 / (dims * num_coeff_per_dim * num_points))
            * (
                DPhi @ B
                - f(t.to(torch.float32), (Phi @ B).to(torch.float32)).to(torch.float64)
            )
            ** 2
            * d
        )
        return torch.sum(error)

    B_tail = newton(
        error, B_init.reshape(batches, -1), f_args=(y_init, f_init), etol=etol
    ).reshape(batches, num_coeff_per_dim - 2, dims)

    torch.set_default_dtype(torch.float32)
    return torch.cat((vmap(l)(B_tail, y_init, f_init), B_tail), dim=1).to(torch.float32)


def pan_int(
    f,
    t_eval,
    y_init,
    num_coeff_per_dim,
    num_points,
    etol_ls=1e-5,
    etol_newton=1e-5,
    callback=None,
    return_B=False,
):
    t_lims = [t_eval[0], t_eval[-1]]

    torch.set_default_dtype(torch.float64)
    f_init = f(t_lims[0], y_init).to(torch.float64)
    batches, dims = y_init.shape

    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    Phi = T_grid(t, num_coeff_per_dim)
    DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim)

    B_init = torch.rand(
        batches,
        num_coeff_per_dim - 2,
        dims,
    )

    # first apply
    B_ls = lst_sq_solver(
        f,
        t_lims,
        y_init,
        num_coeff_per_dim,
        num_points,
        f_init,
        Phi[None],
        DPhi[None],
        B_init=B_init,
    )

    B_newton = newton_solver(
        f,
        t_lims,
        y_init,
        num_coeff_per_dim,
        num_points,
        f_init,
        Phi,
        DPhi,
        # B_init=B_init[:, :-2, :],
        B_init=B_ls[:, :-2, :],
    )

    t_phi = -1 + 2 * (t_eval - t_lims[0]) / (t_lims[1] - t_lims[0])
    Phi_traj = T_grid(t_phi, num_coeff_per_dim)

    traj = Phi_traj @ B_newton
    # transpose to comply with torchdyn
    traj.transpose_(0, 1).to(torch.float32)

    if return_B:
        return traj, B_newton
    else:
        return traj


def make_pan_adjoint(
    f,
    thetas,
    num_coeff_per_dim,
    num_points,
    etol_ls=1e-5,
    etol_newton=1e-9,
    callback=None,
):
    class _PanInt(Function):
        @staticmethod
        def forward(ctx, thetas, y_init, t_eval):
            traj, B = pan_int(
                f,
                t_eval,
                y_init,
                num_coeff_per_dim,
                num_points,
                etol_ls,
                etol_newton,
                callback,
                return_B=True,
            )
            ctx.save_for_backward(t_eval, traj, B)

            return t_eval, traj

        @staticmethod
        def backward(ctx, *grad_output):
            """
            heavily inspired by https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/sensitivity.py#L32
            """
            t_eval, ys, B = ctx.saved_tensors

            num_points, batches, dims = ys.shape

            dLdy = grad_output[1][-1]
            dLdt = grad_output[1]
            theta = torch.cat([p.contiguous().flatten() for p in f.parameters()])

            # pdb.set_trace()

            a_theta_T = torch.zeros_like(theta)
            a_theta_shape = a_theta_T.shape
            # TODO: gradients wrt to t

            t_phi = -1 + 2 * (t_eval - t_eval[0]) / (t_eval[1] - t_eval[0])
            Phi = T_grid(t_eval, num_coeff_per_dim)

            def adjoint_dynamics(t, a_y):
                # set batch as first dimension
                with torch.set_grad_enabled(True):
                    Dys = f(t_eval, ys)

                # pdb.set_trace()
                return torch.autograd.grad(dy, ys, -a_y)

            a_y_traj = pan_int(
                adjoint_dynamics,
                t_eval,
                dLdy,
                num_coeff_per_dim,
                num_points,
                etol_ls,
                etol_newton,
                callback,
            )

    def _pan_int_adjoint(y_init, t_eval):
        return _PanInt.apply(thetas, y_init, t_eval)

    return _pan_int_adjoint
