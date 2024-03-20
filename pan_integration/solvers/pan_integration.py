import torch
from torch import Tensor, tensor, vstack, hstack
from torch.linalg import inv
from ..utils.plotting import VfPlotter, wait
from ..optim import newton
import sys
from torch import pi as PI


def T_grid(num_points, num_coeff_per_dim):
    t = torch.linspace(-1, 1, num_points)[:, None]

    out = torch.empty(num_points, num_coeff_per_dim)
    out[:, [0]] = torch.ones(num_points, 1, dtype=torch.float)
    out[:, [1]] = t

    for i in range(2, num_coeff_per_dim):
        out[:, [i]] = 2 * t * out[:, [i - 1]] - out[:, [i - 2]]

    return out


def U_grid(num_points, num_coeff_per_dim):
    t = torch.linspace(-1, 1, num_points)[:, None]

    out = torch.empty(num_points, num_coeff_per_dim)
    out[:, [0]] = torch.ones(num_points, 1, dtype=torch.float)
    out[:, [1]] = 2 * t

    for i in range(2, num_coeff_per_dim):
        out[:, [i]] = 2 * t * out[:, [i - 1]] - out[:, [i - 2]]

    return out


def _B_init_cond(B_tail, y_init, f_init, Phi, DPhi):
    B_01 = inv(vstack([Phi[0, [0, 1]], DPhi[0, [0, 1]]])) @ (
        vstack([y_init, f_init]) - vstack([Phi[[0], 2:], DPhi[[0], 2:]]) @ B_tail
    )
    return torch.vstack([B_01, B_tail])


def _cheb_phis(num_points, num_coeff_per_dim, t_lims):
    step = t_lims[1] - t_lims[0]
    Phi = T_grid(num_points, num_coeff_per_dim)
    DPhi = (2 / step) * torch.hstack(
        [
            torch.zeros(num_points, 1, dtype=torch.float),
            torch.arange(1, num_coeff_per_dim, dtype=torch.float)
            * U_grid(num_points, num_coeff_per_dim - 1),
        ]
    )
    return Phi, DPhi


def _coarse_euler_lstsq(
    f, y_init, num_solver_steps, t_lims, num_coeff_per_dim, plotter
):
    dims = y_init.shape[0]
    step = t_lims[1] - t_lims[0]
    step_size = step / (num_solver_steps)

    y_eul = torch.zeros(num_solver_steps, dims)
    y_eul[0, :] = y_init
    f_eul = torch.zeros(num_solver_steps, dims)
    f_eul[0, :] = f(y_init)

    # forward euler
    for i in range(1, num_solver_steps):
        y_cur = y_eul[i - 1, :]
        f_eul[i, :] = f(y_cur)
        f_cur = f(y_cur)
        y_eul[i, :] = y_cur + step_size * f_cur
    Phi, DPhi = _cheb_phis(num_solver_steps, num_coeff_per_dim, t_lims)

    inv0 = inv(vstack([Phi[0, [0, 1]], DPhi[0, [0, 1]]]))
    PHI = -vstack([Phi[:, [0, 1]], DPhi[:, [0, 1]]]) @ inv0 @ vstack(
        [Phi[[0], 2:], DPhi[[0], 2:]]
    ) + vstack([Phi[:, 2:], DPhi[:, 2:]])
    Y = vstack([y_eul, f_eul]) - vstack(
        [Phi[:, [0, 1]], DPhi[:, [0, 1]]]
    ) @ inv0 @ vstack([y_init, f_eul[[0], :]])

    B_ls = torch.linalg.lstsq(PHI, Y).solution
    B = _B_init_cond(B_ls, y_init, f(y_init), Phi, DPhi)

    # plotter.approx(Phi @ B, t_lims[0], "orange")
    # wait()

    return B_ls.T.reshape(-1)


def pan_int(
    f,
    y_init: Tensor,
    t_lims: list,
    num_coeff_per_dim: int,
    num_points: int,
    step: float = None,
    etol=1e-5,
    atol= 1e-5,
    callback: callable = None,
    plotter: VfPlotter = None,
    coarse_steps = 5
) -> tuple:
    COUNTER = 0

    if step is None:
        step = t_lims[1] - t_lims[0]

    dims = y_init.shape[0]
    cur_interval = [t_lims[0], t_lims[0] + step]

    approx = torch.tensor([])
    B = torch.rand((num_coeff_per_dim - 2) * dims)
    while True:
        if cur_interval[1] > t_lims[1]:
            cur_interval[1] = t_lims[1]
            step = cur_interval[1] - cur_interval[0]

        Phi, DPhi = _cheb_phis(num_points, num_coeff_per_dim, cur_interval)

        def error_func(B_vec: Tensor) -> tuple:
            with torch.no_grad():
                nonlocal COUNTER
                COUNTER += 1

            B = _B_init_cond(
                B_vec.reshape(dims, num_coeff_per_dim - 2).T,
                y_init,
                f(y_init),
                Phi,
                DPhi,
            )

            approx = Phi @ B
            Dapprox = DPhi @ B

            error = torch.sum((Dapprox - f(approx)) ** 2) / (
                num_points * num_coeff_per_dim
            )
            return error, approx

        B = _coarse_euler_lstsq(f, y_init, coarse_steps, cur_interval, num_coeff_per_dim, plotter)
        COUNTER += coarse_steps

        B = newton(
            error_func,
            B,
            has_aux=True,
            etol=etol,
            atol=atol,
            callback=lambda B: callback(B, y_init, cur_interval)
            if callback is not None
            else None,
        )

        # TODO: make newton return the aux of error_func instead of recalculating
        (error, local_approx) = error_func(B)
        COUNTER -= 1
        y_init = local_approx[-1, :]

        approx = torch.cat((approx, local_approx))
        if cur_interval[1] == t_lims[1]:
            return approx, COUNTER
        else:
            cur_interval = [cur_interval[1], cur_interval[1] + step]
