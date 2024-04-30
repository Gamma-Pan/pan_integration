import torch
from torch import Tensor, vstack, tensor, cat, stack
from torch.linalg import inv
from torch.func import vmap

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


def zero_order_pan_solver(
    f,
    t_lims,
    y_init,
    num_coeff_per_dim,
    num_points,
    f_init=None,
    Phi=None,
    DPhi=None,
    B_init=None,
    max_steps=20,
    etol=1e-5,
    callback=None,
) -> Tensor | Tuple[Tensor, int]:
    device = y_init.device

    t_hat = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    t = t_lims[0] + (t_hat - t_hat[0]) * (t_lims[1] - t_lims[0]) / (
        t_hat[-1] - t_hat[0]
    )

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
            Phi_bT.mT @ (d *
                        # vectorize f along time dimension
                         vmap(f, in_dims=(0, 1))(t_hat, (Phi @ cat((l(B), B), dim=1)))
                         )
            - Phi_bT.mT @ (d * Phi_aT)
        )

        if torch.norm(B - B_prev) < etol:
            break
        if i == max_steps - 1:
            print('max_iters')


    return cat((l(B), B), dim=1)



def pan_int(
    f,
    t_eval,
    y_init,
    num_coeff_per_dim,
    num_points,
    etol_ls=1e-5,
    max_iters_ls=20,
    callback=None,
    f_init=None,
    return_B=False,
):
    t_lims = [t_eval[0], t_eval[-1]]

    device = y_init.device
    batches, dims = y_init.shape
    if f_init is None:
        f_init = f(t_lims[0], y_init)

    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points)).to(device)
    Phi = T_grid(t, num_coeff_per_dim, device)
    DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim, device)

    B_init = torch.rand((batches, num_coeff_per_dim - 2, dims), device=device)

    # first apply
    B_ls = zero_order_pan_solver(
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
        max_steps=max_iters_ls
    )

    t_phi = -1 + 2 * (t_eval - t_lims[0]) / (t_lims[1] - t_lims[0])
    Phi_traj = T_grid(t_phi.to(device), num_coeff_per_dim, device)

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
                torch.abs(t_eval[0] - t_eval[-1]) / (num_points_adjoint-1)
            ) * grads_vec

            # ipdb.set_trace()

            return DL_theta, None, None

    def _pan_int_adjoint(y_init, t_eval):
        return _PanInt.apply(thetas, y_init, t_eval)

    return _pan_int_adjoint
