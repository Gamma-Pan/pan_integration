import torch
from torch import Tensor, vstack, tensor, cat, stack
from torch.linalg import inv
from torch.func import vmap

# from ..optim import newton
from typing import Tuple
from functools import partial

from torch.autograd import Function
import ipdb

# torch.set_default_dtype(torch.float64)
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

    t_hat = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    t = t_lims[0] + (t_hat - t_hat[0]) * (t_lims[1] - t_lims[0]) / (
        t_hat[-1] - t_hat[0]
    )

    batches, dims = y_init.shape

    if f_init is None:
        f_init = f(t[0], y_init)

    if Phi is None or DPhi is None:
        Phi = T_grid(t_hat, num_coeff_per_dim, device)[None]
        DPhi = (
            2
            / (t_lims[1] - t_lims[0])
            * DT_grid(t_hat, num_coeff_per_dim, device)[None]
        )

    # approximating Rieman integral with non uniformly spaced points requires to multiply
    # with the interval lenghts between points
    # TODO: approximate integral with sum using trapezoids instead of rectangles
    d = torch.diff(torch.cat((t_hat, tensor(1, device=device)[None])))[:, None].to(
        device
    )

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
            Phi_bT.mT @ (d * f(t_hat, (Phi @ cat((l(B), B), dim=1))))
            - Phi_bT.mT @ (d * Phi_aT)
        )

        if torch.norm(B - B_prev) < etol:
            break

    return cat((l(B), B), dim=1)


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

    t_hat = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    t = t_lims[0] + (t_hat - t_hat[0]) * (
        (t_lims[1] - t_lims[0]) / (t_hat[-1] - t_hat[0])
    )
    batches, dims = y_init.shape

    if f_init is None:
        f_init = f(t[0], y_init)

    if Phi is None or DPhi is None:
        Phi = T_grid(t_hat, num_coeff_per_dim, device)
        DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t_hat, num_coeff_per_dim, device)

    # approximating Rieman integral with non uniformly spaced points requires to multiply
    # with the interval lenghts between points
    # TODO: approximate integral with sum using trapezoids instead of rectangles
    d = torch.diff(torch.cat((t, tensor(t_lims[1])[None])))[:, None].to(device)

    inv0 = inv(cat((Phi[0:1, [0, 1]], DPhi[0:1, [0, 1]]), dim=0))
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
            * (DPhi @ B - f(t, (Phi @ B))) ** 2
            * d
        )
        return torch.sum(error)

    B_tail = newton(
        error, B_init.reshape(batches, -1), f_args=(y_init, f_init), etol=etol
    ).reshape(batches, num_coeff_per_dim - 2, dims)

    return torch.cat((vmap(l)(B_tail, y_init, f_init), B_tail), dim=1)


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

    device = y_init.device
    f_init = f(t_lims[0], y_init)
    batches, dims = y_init.shape

    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    Phi = T_grid(t, num_coeff_per_dim, device)
    DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim, device)

    B_init = torch.rand((batches, num_coeff_per_dim - 2, dims), device=device)

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
        etol=etol_ls,
    )

    # B_newton = newton_solver(
    #     f,
    #     t_lims,
    #     y_init,
    #     num_coeff_per_dim,
    #     num_points,
    #     f_init,
    #     Phi,
    #     DPhi,
    #     # B_init=B_init[:, :-2, :],
    #     B_init=B_ls[:, 2:, :],
    #     etol=etol_newton,
    # )

    t_phi = -1 + 2 * (t_eval - t_lims[0]) / (t_lims[1] - t_lims[0])
    Phi_traj = T_grid(t_phi, num_coeff_per_dim, device)

    traj = Phi_traj @ B_ls
    # transpose to comply with torchdyn
    traj.transpose_(0, 1)

    if return_B:
        return traj, B_ls
    else:
        return traj


def make_pan_adjoint(
    f,
    thetas,
    num_coeff_per_dim,
    num_points,
    etol_ls=1e-5,
    etol_newton=1e-3,
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
                etol_newton,
                callback,
                return_B=True,
            )
            ctx.save_for_backward(t_eval, traj, B_fwd)

            return t_eval, traj

        @staticmethod
        def backward(ctx, *grad_output):
            t_eval, ys, B_fwd = ctx.saved_tensors

            points, batch_sz, dims = ys.shape

            # dL/dz(T) -> (1xBxD)
            a_y_T = grad_output[1][-1]

            theta = torch.cat(
                [v.contiguous().flatten() for k, v in f.named_parameters()]
            )
            a_theta_sz = torch.numel(theta)
            a_theta_T = torch.zeros(batch_sz, a_theta_sz)

            # DEFINE AUGMENTED SYSTEM OF A_Y AND A_THETA
            # initial state (B,dims + weights_sz)
            A_T = torch.cat([a_y_T, a_theta_T], dim=1)

            def adjoint_dynamics(t, A):
                # get y(t)
                num_points = t.numel()
                a_y = A[..., :dims]
                if num_points == 1:
                    a_y.unsqueeze_(1)

                yt = T_grid(t, num_coeff_per_dim, device=t.device) @ B_fwd

                _, _vjp_func_ay = torch.func.vjp(lambda y: f(t, y), yt)
                (Da_y,) = _vjp_func_ay(a_y)

                def single_Da_theta(y_t, a_y_t):
                    _, _vjp_func_a_theta = torch.func.vjp(
                        lambda x: torch.func.functional_call(f, x, (t, y_t)),
                        {k: v.detach() for k, v in f.named_parameters()},
                    )
                    (Da_theta_sample,) = _vjp_func_a_theta(a_y_t)
                    return Da_theta_sample

                # first map over batches, then over time
                Da_theta = vmap(vmap(single_Da_theta, in_dims=(0, 0)), in_dims=(0, 0))(
                    yt, a_y
                )

                Da_theta = torch.cat(
                    [
                        v.flatten(start_dim=2) if v is not None else torch.zeros(batch_sz,num_points, 1)
                        for v in Da_theta.values()
                    ],
                    dim=2
                )

                DA_T = torch.cat([Da_y, Da_theta], dim=2)
                if num_points ==1:
                    DA_T.squeeze_(1)
                return DA_T

            A_traj = pan_int(
                adjoint_dynamics,
                t_eval.flip(0),
                A_T,
                num_coeff_per_dim,
                num_points,
                etol_ls,
                etol_newton,
                callback,
            )

            DlDtheta = A_traj[-1, a_theta_numel:]

            ipdb.set_trace()

    def _pan_int_adjoint(y_init, t_eval):
        return _PanInt.apply(thetas, y_init, t_eval)

    return _pan_int_adjoint
