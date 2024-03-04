import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable

from matplotlib.animation import PillowWriter
from scipy.integrate import solve_ivp
import torch
from itertools import cycle


mpl.use("TkAgg")
quiver_args = {
    "headwidth": 2,
    "headlength": 1,
    "headaxislength": 1.5,
    "linewidth": 0.1,
    "angles": "xy",
}

stream_kwargs = {
    'color': '#555555',
    'density': 0.7,
    'arrowsize': 0.6,
    'linewidth': 0.6
}


def wait():
    while True:
        if plt.waitforbuttonpress():
            break


class VfPlotter:
    def __init__(
            self,
            f: Callable,
            y_init: torch.Tensor,
            t_init: float,
            grid_definition: tuple = (40, 40),
            existing_axes: plt.Axes = None,
            ax_kwargs: dict = None,
            show=False,
            animation=False
    ):

        self.cycol = cycle('b')
        self.y_init = y_init
        self.t_init = t_init

        if existing_axes is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set(**ax_kwargs)
        else:
            self.ax = existing_axes

        self.ax.plot(*y_init, "ro")
        self.approx_art = None

        self.grid_definition = grid_definition
        self.f = f

        self._plot_vector_field()

        self.animation = animation
        if animation:
            self.writer = PillowWriter(fps=4, metadata={'artist': 'Yiorgos Pan'})
            self.writer.setup(self.fig, 'test.gif', 100)
            self.writer.frame_format = 'png'


        if show:
            wait()

    def _plot_vector_field(self):
        xs = torch.linspace(*self.ax.get_xlim(), self.grid_definition[0])
        ys = torch.linspace(*self.ax.get_ylim(), self.grid_definition[1])

        Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")

        batch = torch.stack((Xs, Ys), dim=-1)
        derivatives = self.f(batch)
        Us, Vs = derivatives.unbind(dim=-1)

        self.ax.streamplot(Xs.numpy().astype(float), Ys.numpy().astype(float),
                           Us.numpy().astype(float), Vs.numpy().astype(float),
                           **stream_kwargs)

    def update(self):
        raise NotImplementedError

    def solve_ivp(
            self, interval, y_init: torch.tensor = None, ivp_kwargs=None, plot_kwargs=None
    ):
        plot_kwargs = plot_kwargs or {}
        ivp_kwargs = ivp_kwargs or {}
        y_init = y_init or self.y_init

        ivp_sol = solve_ivp(
            lambda t, x: self.f(x.astype(float)),
            interval,
            y_init,
            **ivp_kwargs,
        )

        trajectory = ivp_sol.y
        print(f"Method {ivp_kwargs['method']} took {ivp_sol.nfev} function evaluations, y(T) = ({trajectory[0][-1]:.6},{trajectory[1][-1]:.6} )")

        (self.trajectory,) = self.ax.plot(*trajectory, **plot_kwargs, label=f"{ivp_kwargs['method']} - {ivp_sol.nfev} NFE")
        return trajectory[:, -1]

    @torch.no_grad()
    def pol_approx(self, approx, t_init, arrows_every_n: int = 10, **kwargs):
        """
        Plot the trigonometric polynomial approximation of the ode solution
        :param arrows_every_n:
        :param show:
        :return:
        """

        if self.approx_art is None:
            self.approx_art, = self.ax.plot([], [], **kwargs, label="pan")

        if self.t_init != t_init:
            self.t_init = t_init
            self.ax.plot(approx[0, 0], approx[0, 1], "o")
            (self.approx_art,) = self.ax.plot(approx[:, 0], approx[:, 1], **kwargs)
        else:
            self.approx_art.set_xdata(approx[:, 0])
            self.approx_art.set_ydata(approx[:, 1])
            self.approx_art.set_color(next(self.cycol))

    def grab_frame(self):
        self.fig.canvas.draw()
        self.writer.grab_frame()


class LsPlotter:

    def __init__(self, func, plot_res=100, alpha_max=10):
        self.fig, self.ax = plt.subplots()
        self.phi = func
        self.plot_res = plot_res
        self.alpha_max = alpha_max

        self.alpha_line_art, = self.ax.plot([], [], color='#EE2233', label='$\phi(a)$')
        self.curpoint, = self.ax.plot([], [], color='#FF5566', label='$a_i$', marker='o')

    def line_search(self, a_cur):
        a = torch.linspace(0, a_cur + 1, self.plot_res)
        phi_a = torch.stack([self.phi(a) for a in a])
        self.alpha_line_art.set_xdata(a)

        self.alpha_line_art.set_ydata(phi_a)
        self.ax.set_xlim(-0.1, a_cur + 1)
        self.ax.set_ylim([0.9 * torch.min(phi_a), self.phi(a_cur + 1) * 1.1])

        self.curpoint.set_xdata([a_cur])
        self.curpoint.set_ydata([self.phi(a_cur)])
        wait()

    def zoom(self):
        raise NotImplementedError

    def close_all(self):
        plt.close(self.fig)
