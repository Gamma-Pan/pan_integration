import torch
from torch import Tensor, tensor, vstack, hstack
from torch.linalg import inv
from ..optim import newton


def T_grid(num_points, num_coeff_per_dim):
    # t = torch.linspace(-1, 1, num_points)[:, None]
    t = -torch.cos(
        torch.pi * (torch.arange(num_points, dtype=torch.float) / (num_points - 1))
    )[:, None]

    out = torch.empty(num_points, num_coeff_per_dim)
    out[:, [0]] = torch.ones(num_points, 1, dtype=torch.float)
    out[:, [1]] = t

    for i in range(2, num_coeff_per_dim):
        out[:, [i]] = 2 * t * out[:, [i - 1]] - out[:, [i - 2]]

    return out


def U_grid(num_points, num_coeff_per_dim):
    # t = torch.linspace(-1, 1, num_points)[:, None]
    t = -torch.cos(
        torch.pi * (torch.arange(num_points, dtype=torch.float) / (num_points - 1))

    )[:, None]
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


def _coarse_euler_init(f, y_init, num_solver_steps, t_lims, num_coeff_per_dim):
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

    B_ls = torch.linalg.lstsq(PHI.to(torch.double), Y.to(torch.double)).solution.to(torch.float)
    B = _B_init_cond(B_ls, y_init, f_eul[0, :], Phi, DPhi)
    print(f"{y_eul[0,:].tolist()} - {(Phi@B)[0,:].tolist()}")
    return B_ls.T.reshape(-1)


def lst_sq_solver(
    f,
    y_init,
    f_init,
    t_lims,
    num_coeff_per_dim,
    num_points,
    max_steps=20,
    etol=1e-5,
    coarse_steps=5,
    return_nfe=False,
    callback=None,
):
    Phi, DPhi = _cheb_phis(num_points, num_coeff_per_dim, t_lims)

    nfe = 0

    dims = y_init.shape[0]
    B = (
        _coarse_euler_init(f, y_init, coarse_steps, t_lims, num_coeff_per_dim)
        .reshape(dims, num_coeff_per_dim - 2)
        .T
    )

    nfe += 1

    # TODO: figure out how to make this calculations into a function,
    #  they are also used in euler solution and in main loop
    inv0 = inv(vstack((Phi[0, [0, 1]], DPhi[0, [0, 1]])))
    Phi_a = DPhi[:, [0, 1]] @ inv0 @ vstack((y_init, f_init))
    Phi_b = (
        -DPhi[:, [0, 1]] @ inv0 @ vstack((Phi[[0], 2:], DPhi[[0], 2:])) + DPhi[:, 2:]
    )
    l = lambda B: inv0 @ (
        vstack((y_init, f_init)) - vstack((Phi[[0], 2:], DPhi[[0], 2:])) @ B
    )

    Q = inv(Phi_b.T @ Phi_b)
    for i in range(max_steps):
        if callback is not None:
            callback(B.T.reshape(-1))

        B_prev = B
        B = Q @ (Phi_b.T @ f(Phi @ vstack((l(B), B))) - Phi_b.T @ Phi_a)

        nfe += 1
        if torch.norm(B - B_prev) < etol:
            break

    sol = Phi @ vstack((l(B), B))

    if return_nfe:
        return B.T.reshape(-1), sol, nfe
    else:
        return (
            B.T.reshape(-1),
            sol,
        )


def pan_int(
    f,
    y_init: Tensor,
    t_lims: list,
    num_coeff_per_dim_newton: int,
    num_coeff_per_dim_ls: int,
    num_points: int,
    step: float = None,
    etol_newton=1e-5,
    etol_lstsq=1e-5,
    callback: callable = None,
    coarse_steps=5,
) -> tuple:
    COUNTER = 0

    if step is None:
        step = t_lims[1] - t_lims[0]

    dims = y_init.shape[0]
    cur_interval = [t_lims[0], t_lims[0] + step]

    approx = torch.tensor([])

    while True:
        if cur_interval[1] > t_lims[1]:
            cur_interval[1] = t_lims[1]
            step = cur_interval[1] - cur_interval[0]

        f_init = f(y_init)

        # start with the least squares solution
        B, sol, ls_nfe = lst_sq_solver(
            f,
            y_init,
            f_init,
            cur_interval,
            num_coeff_per_dim_ls,
            num_points,
            etol=etol_lstsq,
            coarse_steps=coarse_steps,
            callback=lambda B: callback(B, y_init, cur_interval)
            if callback is not None
            else None,
            return_nfe=True,
        )

        # switch to newton for refinement
        Phi, DPhi = _cheb_phis(num_points, num_coeff_per_dim_newton, cur_interval)

        inv0 = inv(vstack((Phi[0, [0, 1]], DPhi[0, [0, 1]])))
        Phi_a = Phi[:, [0, 1]] @ inv0 @ vstack((y_init, f_init))
        Phi_b = (
            -Phi[:, [0, 1]] @ inv0 @ vstack((Phi[[0], 2:], DPhi[[0], 2:])) + Phi[:, 2:]
        )
        B = torch.linalg.lstsq(Phi_b, sol - Phi_a)[0].T.reshape(-1)

        def error_func(B_vec: Tensor) -> tuple:
            with torch.no_grad():
                nonlocal COUNTER
                COUNTER += 1

            B = _B_init_cond(
                B_vec.reshape(dims, num_coeff_per_dim_newton - 2).T,
                y_init,
                f(y_init),
                Phi,
                DPhi,
            )

            approx = Phi @ B
            Dapprox = DPhi @ B

            error = torch.sum((Dapprox - f(approx)) ** 2) / (
                num_points * num_coeff_per_dim_newton
            )
            return error, approx

        B = newton(
            error_func,
            B,
            has_aux=True,
            etol=etol_newton,
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
            return approx, ( ls_nfe,COUNTER)
        else:
            cur_interval = [cur_interval[1], cur_interval[1] + step]
