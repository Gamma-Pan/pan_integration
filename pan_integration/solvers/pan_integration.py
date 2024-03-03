from typing import Iterable

import torch
from torch import tensor, Tensor, cos, sin
from ..utils.plotting import VfPlotter
from ..optim import newton

from torch import pi as PI


def pan_int(
        f,
        y_init: Tensor,
        t_lims: list,
        num_coeff_per_dim: int,
        num_points: int,
        step: float = None,
        etol=1e-3,
        callback: callable = None,
) -> Tensor:
    """

    :param f:
    :param y_init:
    :param t_lims:
    :param num_coeff_per_dim:
    :param num_points:
    :param step:
    :param etol:
    :param plot:
    :param plotter_kwargs:
    :param callback: a function that's called after each newton iteration
    :return:
    """

    if step is None:
        step = t_lims[1] - t_lims[0]

    dims = y_init.shape[0]
    cur_interval = [t_lims[0], t_lims[0] + step]

    freqs = torch.arange(num_coeff_per_dim // 2, dtype=torch.float)[None, :]

    # two rows of b are not learnable but defined from y_init
    # one row is defined from f(y_init)
    B = torch.rand((num_coeff_per_dim-3)*dims)

    cur_interval = [t_lims[0], t_lims[0] + step]
    approx = torch.tensor([])
    while True:

        if cur_interval[1] > t_lims[1]:
            cur_interval[1] = t_lims[1]

        t_points = torch.linspace(
            cur_interval[0], cur_interval[1], num_points, dtype=torch.float
        )[:, None]
        # broadcast product is NxM
        grid = (t_points * freqs)

        # caclulate the polynomial time variables and their derivatives
        Phi_cT = cos(grid)
        Phi_sT = sin(grid)
        D_Phi_cT = -freqs * sin(grid)
        D_Phi_sT = freqs * cos(grid)

        def error_func(B_vec: Tensor) -> tuple:
            t_0 = torch.tensor(cur_interval[0])
            # CONVERT VECTOR BACK TO MATRICES

            #  the first half of B comprises the Bc matrix minus the first row
            Bc_m1 = B_vec[:(num_coeff_per_dim // 2 - 1) * dims].reshape(num_coeff_per_dim // 2 - 1, dims)
            # the rest comprises the Bs matrix minus the first two rows
            Bs_m2 = B_vec[(num_coeff_per_dim // 2 - 1) * dims:].reshape(num_coeff_per_dim // 2 - 2, dims)

            # calculate the 1-index row of Bs
            Bs_1 = ((f(y_init[None])
                     + sin(t_0) * Bc_m1[0:1, :]
                     - D_Phi_cT[0, 2:] @ Bc_m1[1:, :]
                     - D_Phi_sT[0, 2:] @ Bs_m2)
                    / cos(t_0))

            # calculate the 1-index row of Bs
            Bs_m1 = torch.vstack((Bs_1, Bs_m2))

            # calculate the first row of Bc
            Bc_0 = y_init[None] - Phi_cT[0:1, 1:] @ Bc_m1 - Phi_sT[0:1, 1:] @ Bs_m1
            Bc = torch.vstack((Bc_0, Bc_m1))

            # the first row Bs doesn't contribute
            Bs = torch.vstack((torch.zeros(1, dims), Bs_m1))




            approx = Phi_cT @ Bc + Phi_sT @ Bs

            # f treats each row independently (rows = batch dimension)
            f_approx = f(approx)  # <--- PARALLELIZE HERE
            D_approx = D_Phi_cT @ Bc + D_Phi_sT @ Bs

            error = torch.sum((f_approx - D_approx) ** 2) / (num_points*num_coeff_per_dim)

            return error, approx

        # initialize B to linear approximation of solution
        t_ls = torch.linspace(*cur_interval, num_coeff_per_dim)[:,None]
        grid_ls = t_ls * freqs
        Phi_cT_ls = cos(grid_ls)
        Phi_sT_ls = sin(grid_ls)
        Phi_ls = torch.hstack( (Phi_cT_ls, Phi_sT_ls) )
        y_ls = y_init + f(y_init[None])*t_ls
        B_init = torch.linalg.lstsq( Phi_ls, y_ls ).solution
        mask = torch.ones_like(B_init)
        mask[0,:] = 0
        mask[num_coeff_per_dim//2: num_coeff_per_dim//2+2, :] = 0
        B = B_init[mask.bool()]

        B = newton(error_func, B, has_aux=True, tol=etol,
                   callback=lambda x: callback(x, y_init, cur_interval, num_coeff_per_dim, dims))

        # TODO: make newton return the aux of error_func instead of recalculating
        (error, local_approx) = error_func(B)
        y_init = local_approx[-1, :]

        torch.cat((approx, local_approx))
        if cur_interval[1] == t_lims[1]:
            return approx
        else:
            cur_interval = [cur_interval[1], cur_interval[1]+step]