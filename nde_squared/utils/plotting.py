import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import solve_ivp
import torch
import numpy as np
from torch import tensor

mpl.use("TkAgg")

quiver_args = {
    "headwidth": 1,
    "headlength": 1,
    "headaxislength": 1,
    "linewidth": 0.1,
    "angles": "xy",
    # "scale": 0.1,
}


class VfPlotter:
    def __init__(
        self,
        f: Callable,
        y_init: torch.Tensor = None,
        grid_definition: tuple = (40, 40),
        existing_axes: plt.Axes = None,
        fig_kwargs: dict = None,
        ax_kwargs: dict = None,
        ivp_kwargs: dict = None,
    ):
        plt.ion()

        if existing_axes is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = existing_axes

        self.grid_definition = grid_definition
        self.f = f

        (self.approx_line,) = self.ax.plot(
            [], [], "o-", color="red", label="approximation", markersize=4
        )
        self.f_points_arrows = None
        self.Phi_point_arrows = None

        padding = 0.1
        # if an initial point is specified set ax limits to contain it
        if y_init is not None:
            self.y_init = y_init
            self.f_init = f(y_init)
            mins, maxs = self.ivp(y_init=y_init, **ivp_kwargs or dict())
            self.ax.set_xlim((mins[0] - padding), maxs[0] + padding)
            self.ax.set_ylim((mins[1] - padding), maxs[1] + padding)

        self._grid_values()

        self.fig.canvas.draw()

    @torch.no_grad()
    def _grid_values(self):
        xs = torch.linspace(*self.ax.get_xlim(), self.grid_definition[0])
        ys = torch.linspace(*self.ax.get_ylim(), self.grid_definition[1])

        Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")
        # batch of all pairs, stack in last dimension so that it's the fastest changing and reshape keeps pairs intact
        z = torch.stack((Xs, Ys), dim=-1).reshape(-1, 2)
        dzdt = self.f(z)
        Us, Vs = dzdt.reshape(
            self.grid_definition[0], self.grid_definition[1], 2
        ).unbind(-1)

        # self.ax.streamplot(Xs.numpy(), Ys.numpy(), Us.numpy(), Vs.numpy())
        self.ax.quiver(Xs, Ys, Us, Vs, angles="xy")

    def update(self):
        raise NotImplementedError

    def ivp(self, y_init: torch.tensor, **ivp_kwargs):
        ivp_kwargs["method"] = ivp_kwargs.get("method") or "LSODA"
        sol = solve_ivp(
            lambda t, x: self.f(torch.tensor(np.array([x], dtype=np.float32))),
            [0, 1],
            y_init.squeeze(),
            **ivp_kwargs
        )
        print(sol.nfev)
        trajectory = tensor(sol.y)
        self.ax.plot(
            trajectory[0, :], trajectory[1, :], "-o", color="forestgreen", markersize=4
        )
        self.ax.plot(trajectory[0, 0], trajectory[1, 0], "o", color="forestgreen")

        return torch.min(trajectory, dim=1)[0], torch.max(trajectory, dim=1)[0]

    @torch.no_grad()
    def approximation(
        self,
        traj,
        d_traj: torch.Tensor = None,
        f: Callable = None,
        arrows_every_n: int = 10,
    ):
        n = arrows_every_n

        y0, y1 = traj.split(1, dim=1)
        self.approx_line.set_data(y0, y1)

        if self.f_points_arrows is not None:
            self.f_points_arrows.remove()

        if f is not None:
            f0, f1 = f(traj).split(1, dim=1)
            self.f_points_arrows = self.ax.quiver(
                y0[::n],
                y1[::n],
                f0[::n],
                f1[::n],
                color="dimgray",
                zorder=10,
                **quiver_args
            )

        if self.Phi_point_arrows is not None:
            self.Phi_point_arrows.remove()

        if d_traj is not None:
            dy0, dy1 = d_traj.split(1, dim=1)
            self.Phi_point_arrows = self.ax.quiver(
                y0[::n],
                y1[::n],
                dy0[::n],
                dy1[::n],
                color="blue",
                zorder=11,
                **quiver_args
            )

        while True:
            if plt.waitforbuttonpress():
                break
