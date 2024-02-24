import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import solve_ivp
import torch
import numpy as np
from torch import tensor, Tensor

mpl.use("TkAgg")

quiver_args = {
    "headwidth": 1,
    "headlength": 1,
    "headaxislength": 1,
    "linewidth": 0.1,
    "angles": "xy",
}


def wait():
    while True:
        if plt.waitforbuttonpress():
            break


class VfPlotter:
    def __init__(
        self,
        f: Callable,
        y_init: torch.Tensor = None,
        grid_definition: tuple = (40, 40),
        existing_axes: plt.Axes = None,
        ax_kwargs: dict = None,
        show=False,
    ):
        if existing_axes is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set(**ax_kwargs)
        else:
            self.ax = existing_axes

        self.grid_definition = grid_definition
        self.f = f

        self._plot_vector_field()

        if show:
            wait()

    @torch.no_grad()
    def _plot_vector_field(self):
        xs = torch.linspace(*self.ax.get_xlim(), self.grid_definition[0])
        ys = torch.linspace(*self.ax.get_ylim(), self.grid_definition[1])

        Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")

        batch = torch.stack((Xs, Ys), dim=-1)
        derivatives = self.f(batch)
        Us, Vs = derivatives.unbind(dim=-1)

        self.ax.quiver(Xs, Ys, Us, Vs, **quiver_args)

    def update(self):
        raise NotImplementedError

    def solve_ivp(
        self, y_init: torch.tensor, ivp_kwargs, plot_kwargs=None
    ):
        plot_kwargs = plot_kwargs or {}

        ivp_sol = solve_ivp(
            lambda t, x: self.f(x),
            [0, 200],
            y_init,
            **ivp_kwargs,
        )

        trajectory = ivp_sol.y
        print(f"Method {ivp_kwargs['method']} took {ivp_sol.nfev} function evaluations")

        (self.trajectory,) = self.ax.plot(*trajectory, **plot_kwargs)

    def pol_approx(
        self,
        approx,
        time,
        arrows_every_n: int = 10,
        show=True,
        **kwargs
    ):
        """
        Plot the trigonometric polynomial approximation of the ode solution
        :param arrows_every_n:
        :param show:
        :return:
        """

        if time != self.time:
            self.time = time
            self.ax.plot(approx[0,0],approx[0,0], 'o')
            self.approx_art, = self.ax.plot(*approx, **kwargs)
        else:
            self.approx_art.set_data(*approx)

        if show:
            wait()



