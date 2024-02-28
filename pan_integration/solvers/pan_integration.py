from typing import Iterable

import torch
from torch import tensor, Tensor, cos, sin
from ..utils.plotting import VfPlotter
from ..optim import newton


def pan_int(
        f,
        y_init: Tensor,
        t_lims: list,
        num_coeff_per_dim: int,
        num_points: int,
        step: float = None,
        etol=1e-3,
        callback: callable = None,
) -> tuple[Tensor, Tensor]:
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
    last_step = False
    # two rows of b are not learnable but defined from y_init
    B_size = (num_coeff_per_dim - 2) * dims
    B = torch.rand(B_size)
    while True:
        if cur_interval[1] >= t_lims[1]:
            cur_interval[1] = t_lims[1]
            last_step = True

        t_points = torch.linspace(
            cur_interval[0], cur_interval[1], num_points, dtype=torch.float
        )[:, None]
        # broadcast product is NxM
        grid = t_points * freqs

        # caclulate the polynomial time variables and their derivatives
        Phi_cT = cos(grid)
        Phi_sT = sin(grid)
        D_Phi_cT = -grid * sin(grid)
        D_Phi_sT = grid * cos(grid)

        def error_func(B_vec: Tensor, y_init: tensor) -> tuple:
            # convert vector back to matrices
            B_c = B_vec[: B_size // 2].reshape(num_coeff_per_dim // 2 - 1, dims)
            B_s = B_vec[B_size // 2:].reshape(num_coeff_per_dim // 2 - 1, dims)

            # calculate the first row of B_c
            B_c0 = y_init[None] - Phi_cT[0:1, 1:] @ B_c - Phi_sT[0:1, 1:] @ B_s
            B_c = torch.vstack((B_c0, B_c))

            # the first row B_s doesn't contribute
            B_s = torch.vstack((torch.zeros(1, dims), B_s))

            approx = Phi_cT @ B_c + Phi_sT @ B_s

            # f treats each row independently (rows = batch dimension)
            f_approx = f(approx)  # <--- PARALLELIZE HERE
            D_approx = D_Phi_cT @ B_c + D_Phi_sT @ B_s

            error = torch.sum((f_approx - D_approx) ** 2)

            return error, approx

        B = newton(error_func, B, f_args=(y_init,), has_aux=True, tol=etol,
                   callback=lambda x: callback(x, Phi_cT, Phi_sT, y_init, cur_interval, num_coeff_per_dim, dims)),

        _, approx = error_func(B, y_init)
        y_init = approx[-1, :]

        cur_interval = [cur_interval[-1], cur_interval[-1] + step]

        if last_step or cur_interval[-1] == t_lims[-1]:
            return approx
