import torch
from torch import Tensor, vstack, tensor, cat, stack
from torch.linalg import inv
from ..optim import newton
from typing import Tuple

# torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

def T_grid(num_points, num_coeff_per_dim, include_end=False):
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    if include_end:
        t = torch.cat((t, tensor([1.0])))
    t = t[:, None]

    out = torch.empty(num_points + include_end, num_coeff_per_dim)
    out[:, [0]] = torch.ones(
        num_points + include_end,
        1,
    )
    out[:, [1]] = t

    for i in range(2, num_coeff_per_dim):
        out[:, [i]] = 2 * t * out[:, [i - 1]] - out[:, [i - 2]]

    return out


def U_grid(num_points, num_coeff_per_dim, include_end=False):
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    if include_end:
        t = torch.cat((t, tensor([1.0])))
    t = t[:, None]
    out = torch.empty(
        num_points + include_end,
        num_coeff_per_dim,
    )
    out[:, [0]] = torch.ones(
        num_points + include_end,
        1,
    )
    out[:, [1]] = 2 * t

    for i in range(2, num_coeff_per_dim):
        out[:, [i]] = 2 * t * out[:, [i - 1]] - out[:, [i - 2]]

    return out


def _cheb_phis(num_points, num_coeff_per_dim, t_lims, include_end=False,device=torch.device("cpu")):
    step = t_lims[1] - t_lims[0]
    Phi = T_grid(num_points, num_coeff_per_dim, include_end)
    DPhi = (2 / step) * torch.hstack(
        [
            torch.zeros(num_points + include_end, 1, dtype=torch.float),
            torch.arange(1, num_coeff_per_dim, dtype=torch.float)
            * U_grid(num_points, num_coeff_per_dim - 1, include_end),
        ]
    )
    return Phi[None].to(device), DPhi[None].to(device)


def _B_init_cond(B_tail, y_init, f_init, Phi, DPhi):
    inv0 = inv(stack((Phi[:, 0, [0, 1]], DPhi[:, 0, [0, 1]]), dim=1))

    l = lambda B: inv0 @ (
        stack((y_init, f_init), dim=1)
        - cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1) @ B_tail
    )

    return torch.cat([l(B_tail), B_tail], dim=1)


@torch.no_grad()
def _coarse_euler_init(f, y_init, num_solver_steps, t_lims, num_coeff_per_dim):
    dims = y_init.shape[0]
    step = t_lims[1] - t_lims[0]
    step_size = step / num_solver_steps
    device = y_init.device

    y_eul = torch.zeros(num_solver_steps, dims).to(device)
    y_eul[0, :] = y_init
    f_eul = torch.zeros(num_solver_steps, dims).to(device)
    f_eul[0, :] = f(y_init)

    # forward euler
    for i in range(1, num_solver_steps):
        y_cur = y_eul[i - 1, :]
        f_cur = f(y_cur)
        f_eul[i, :] = f_cur
        y_eul[i, :] = y_cur + step_size * f_cur

    Phi, DPhi = _cheb_phis(num_solver_steps, num_coeff_per_dim, t_lims)

    inv0 = inv(vstack((Phi[0, [0, 1]], DPhi[0, [0, 1]])))
    PHI = -vstack([Phi[:, [0, 1]], DPhi[:, [0, 1]]]) @ inv0 @ vstack(
        [Phi[[0], 2:], DPhi[[0], 2:]]
    ) + vstack([Phi[:, 2:], DPhi[:, 2:]])

    Y = vstack([y_eul, f_eul]) - vstack(
        [Phi[:, [0, 1]], DPhi[:, [0, 1]]]
    ) @ inv0 @ vstack([y_init, f_eul[[0], :]])

    B_ls = torch.linalg.lstsq(PHI, Y, driver="gels").solution
    return B_ls.T.reshape(-1)


def lst_sq_solver(
    f,
    y_init,
    f_init,
    t_lims,
    num_coeff_per_dim,
    num_points,
    Phi=None,
    DPhi=None,
    max_steps=50,
    etol=1e-5,
    coarse_steps=5,
    return_nfe=False,
    init="random",
    callback=None,
) -> Tensor | Tuple[Tensor, int]:
    device = y_init.device
    if Phi is None or DPhi is None:
        Phi, DPhi = _cheb_phis(num_points, num_coeff_per_dim, t_lims)

    nfe = 0

    batches, dims = y_init.shape

    if init == "euler":
        B = (
            _coarse_euler_init(f, y_init, coarse_steps, t_lims, num_coeff_per_dim)
            .reshape(dims, num_coeff_per_dim - 2)
            .T
        )
        nfe += coarse_steps
    elif init == "random":
        B = torch.rand(batches, num_coeff_per_dim - 2, dims, device=device)
    else:
        raise "init must be either 'euler' or 'random'"

    # TODO: figure out how to make this calculations into a function,
    #  they are also used in euler solution

    # integral interval lengths for non-uniform sampling
    t = -torch.cos(torch.pi * (torch.arange(num_points) / num_points))
    d = t_lims[1] * torch.diff(torch.cat((t, tensor([1.0]))))[:, None]

    inv0 = inv(stack((Phi[:, 0, [0, 1]], DPhi[:, 0, [0, 1]]), dim=1))
    Phi_aT = DPhi[:, :, [0, 1]] @ inv0 @ stack((y_init, f_init), dim=1)
    Phi_bT = (
        -DPhi[:, :, [0, 1]] @ inv0 @ cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1)
        + DPhi[:, :, 2:]
    )
    l = lambda B: inv0 @ (
        stack((y_init, f_init), dim=1)
        - cat((Phi[:, [0], 2:], DPhi[:, [0], 2:]), dim=1) @ B
    )

    Q = inv(Phi_bT.mT @ (d * Phi_bT))
    # MAIN LOOP
    for i in range(max_steps):
        if callback is not None:
            callback(B.T.reshape(-1).cpu())

        B_prev = B
        # B update
        B = Q @ (
            Phi_bT.mT @ (d * f(0, Phi @ cat((l(B), B), dim=1)))
            - Phi_bT.mT @ (d * Phi_aT)
        )

        nfe += 1
        if torch.norm(B - B_prev) < etol:
            break

    if return_nfe:
        return B.mT.reshape(-1), nfe
    else:
        return (B.mT.reshape(-1),)


def pan_int(
    f,
    y_init: Tensor,
    t_lims: list,
    num_coeff_per_dim: int = 20,
    num_points: int = 50,
    step: float = None,
    callback: callable = None,
    ls_kwargs=None,
    newton_kwargs=None,
) -> tuple:
    """
    Integrate an ODE numerically by optimizing a polynomial approximation solution.

    :param f: the batched dynamics fuction: dydt = f(t)
    :param y_init: the initial state
    :param t_lims: limits of integration
    :param num_coeff_per_dim: how many coefficients per dimension to use for the approximation
    :param num_points: at how many points to calculate the MSE
    :param step: the step to split the interval to sub-intervals
    :param callback: a function to call after each iteration of newton or ls
    :param ls_kwargs: kwargs for the ls subrourine
    :param newton_kwargs:  kwargs for the newton subroutine
    :return:
    """
    if ls_kwargs is None:
        ls_kwargs = {}
    if newton_kwargs is None:
        newton_kwargs = {}
    if "callback" in ls_kwargs.keys():
        del ls_kwargs["callback"]
    if "callback" in newton_kwargs.keys():
        del newton_kwargs["callback"]

    ls_kwargs["return_nfe"] = True
    newton_kwargs["has_aux"] = True

    device = y_init.device

    newton_nfe = 0
    ls_nfe_total = 0

    if step is None:
        step = t_lims[1] - t_lims[0]

    dims = y_init.shape[0]
    cur_interval = [t_lims[0], t_lims[0] + step]

    approx = torch.tensor([], device=device)

    while True:
        if cur_interval[1] > t_lims[1]:
            cur_interval[1] = t_lims[1]
            step = cur_interval[1] - cur_interval[0]

        f_init = f(y_init)
        ls_nfe_total += 1

        Phi, DPhi = _cheb_phis(num_points, num_coeff_per_dim, cur_interval, device)

        # START WITH THE LEAST SQUARES SOLUTION
        B, sol, ls_nfe = lst_sq_solver(
            f,
            y_init,
            f_init,
            cur_interval,
            num_coeff_per_dim,
            num_points,
            callback=lambda B: callback(B, y_init, cur_interval)
            if callback is not None
            else None,
            Phi=Phi,
            DPhi=DPhi,
            **ls_kwargs
        )
        ls_nfe_total += ls_nfe

        # SWITCH TO NEWTON FOR REFINEMENT
        def error_func(B_vec: Tensor) -> tuple:
            with torch.no_grad():
                nonlocal newton_nfe
                newton_nfe += 1

            B = _B_init_cond(
                B_vec.reshape(dims, num_coeff_per_dim - 2).T,
                y_init,
                f_init,
                Phi,
                DPhi,
            )

            approx = Phi @ B
            Dapprox = DPhi @ B

            error = torch.sum((Dapprox - f(approx)) ** 2) / (
                num_points * num_coeff_per_dim
            )
            return error, approx

        B, aux = newton(
            error_func,
            B,
            callback=lambda B: callback(B, y_init, cur_interval)
            if callback is not None
            else None,
            **newton_kwargs
        )

        local_approx = aux[0]
        newton_nfe -= 1
        y_init = local_approx[-1, :]

        approx = torch.cat((approx, local_approx))
        if cur_interval[1] >= t_lims[1]:
            return approx, (ls_nfe_total, newton_nfe)
        else:
            cur_interval = [cur_interval[1], cur_interval[1] + step]
