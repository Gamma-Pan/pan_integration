from typing import Callable

import torch
from torch import linspace, meshgrid, tensor, Tensor, cos, sin
from pan_integration.utils.plotting import wait
from pan_integration.optim import newton

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg")


def rosenbrock(batch) -> Tensor:
    a = 1
    b = 100

    x = batch[..., 0]
    y = batch[..., 1]

    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def quadratic_gen(n) -> Callable:
    L = torch.diag(torch.rand(n)*5)
    Q = torch.rand(n,n)*5

    print(torch.diagonal(L))
    print(torch.sort(torch.real(torch.linalg.eigvals(Q @ L @ Q.T )  ))[0])

    def f(batch) -> Tensor:
       x = batch[:,None]
       res = x.T @ (Q @ L @ Q.T) @ x
       return res.squeeze()

    return f

if __name__ == "__main__":

    f = rosenbrock
    grid_def = (100, 100)
    y1_lims = [0.4, 1.6]
    y2_lims = [0.4, 1.6]

    fig, ax = plt.subplots()
    y1s = linspace(y1_lims[0], y1_lims[1], grid_def[0])
    y2s = linspace(y2_lims[0], y2_lims[1], grid_def[1])
    y1_grid, y2_grid = meshgrid(y1s, y2s, indexing="xy")
    grid = torch.stack((y1_grid, y2_grid), dim=-1)
    z_grid = f(grid)

    fig_3d, ax_3d = plt.subplots(
        subplot_kw={"projection": "3d", "computed_zorder": False}
    )
    ax_3d.plot_surface(y1_grid, y2_grid, z_grid.squeeze())
    ax_3d.set_xlim(y1_lims[0], y1_lims[1])
    ax_3d.set_ylim(y2_lims[0], y2_lims[1])

    b_init = torch.tensor([0.45, 0.45])
    b_min = newton(f, b_init)
    print(b_min)
    ax_3d.plot3D(b_min[0], b_min[1], f(b_min))

