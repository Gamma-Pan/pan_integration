import torch
from torch import Tensor, vstack, tensor, cat, stack
from torch.linalg import inv
from torch.func import vmap
from ..optim import newton
from typing import Tuple

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

    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    batches, dims = y_init.shape

    if f_init is None:
        f_init = f(t_lims[0], y_init)

    if Phi is None or DPhi is None:
        Phi = T_grid(t, num_coeff_per_dim)[None]
        DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim)[None]

    if B_init is None:
        B = torch.rand(batches, num_coeff_per_dim - 2, dims)
    else:
        B = B_init

    # approximating Rieman integral with non uniformly spaced points requires to multiply
    # with the interval lenghts between points
    # TODO: approximate integral with sum using trapezoids instead of rectangles
    d = t_lims[1] * torch.diff(torch.cat((t, tensor([1.0]))))[:, None].to(device)

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
    for i in range(max_steps):
        if callback is not None:
            callback(cat((l(B), B), dim=1))

        B_prev = B
        # B update
        B = Q @ (
            Phi_bT.mT @ (d * f(t, Phi @ cat((l(B), B), dim=1)))
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

    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    batches, dims = y_init.shape

    if f_init is None:
        f_init = f(t_lims[0], y_init)

    if Phi is None or DPhi is None:
        Phi = T_grid(t, num_coeff_per_dim)
        DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff_per_dim)

    # approximating Rieman integral with non uniformly spaced points requires to multiply
    # with the interval lenghts between points
    # TODO: approximate integral with sum using trapezoids instead of rectangles
    d = t_lims[1] * torch.diff(torch.cat((t, tensor([1.0]))))[:, None].to(device)

    inv0 = inv(cat((Phi[0:1, [0, 1]], DPhi[0:1, [0, 1]]), dim=0))
    l = lambda B, y_init, f_init: inv0 @ (
        stack((y_init, f_init), dim=0) - cat((Phi[0:1, 2:], DPhi[0:1, 2:]), dim=0) @ B
    )

    # define error for one sample as a function of B vectorised
    def error(B_vec, y_init, f_init):
        B_tail = B_vec.reshape(num_coeff_per_dim - 2, dims)
        B_head = l(B_tail, y_init, f_init)

        B = torch.cat((B_head, B_tail), dim=0)

        error = (DPhi @ B - f(t, Phi @ B)) ** 2 * d
        return torch.sum(error)

    B_tail = newton(error, B_init, f_args=(y_init, f_init)).reshape(
        batches, num_coeff_per_dim - 2, dims
    )

    return torch.cat((vmap(l)(B_tail, y_init, f_init), B_tail), dim=1)
