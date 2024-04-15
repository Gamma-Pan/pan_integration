import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
from itertools import cycle

from matplotlib.animation import PillowWriter
import torch
from torch import max, min, tensor, abs
from torch.linalg import inv

from torchdyn.numerics import odeint


mpl.use("TkAgg")

quiver_args = {
    "headwidth": 2,
    "headlength": 1,
    "headaxislength": 1.5,
    "linewidth": 0.1,
    "angles": "xy",
}

stream_kwargs = {
    "color": "#55555533",
    "density": 0.5,
    "arrowsize": 0.8,
    "linewidth": 0.8,
}


def wait():
    while True:
        if plt.waitforbuttonpress():
            break


class VfPlotter:
    def __init__(
        self,
        f: Callable,
        grid_definition: tuple = (40, 40),
        existing_axes: plt.Axes = None,
        ax_kwargs: dict = None,
        animation=False,
        text=False,
        queue_size=5,
    ):
        if existing_axes is None:
            self.fig, self.ax = plt.subplots()
            if ax_kwargs is not None:
                self.ax.set(**ax_kwargs)
        else:
            self.ax = existing_axes

        self.D_quiver = None
        self.f_quiver = None

        self.t_init = None
        self.grid_definition = grid_definition
        self.f = f

        if animation:
            self.writer = PillowWriter(fps=4, metadata={"artist": "Yiorgos Pan"})
            self.writer.setup(self.fig, "test.gif", 100)
            self.writer.frame_format = "png"
        self.animation = animation

    def _plot_vector_field(self, trajectories):
        padding = 0.2
        xmax = torch.max(trajectories) + padding
        xmin = torch.min(trajectories) - padding
        ymax = torch.max(trajectories) + padding
        ymin = torch.min(trajectories) - padding

        self.ax.set_xlim(xmin , xmax )
        self.ax.set_ylim(ymin , ymax )

        xs = torch.linspace(float(xmin), float(xmax), self.grid_definition[0])
        ys = torch.linspace(float(ymin), float(ymax), self.grid_definition[1])

        Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")

        batch = torch.stack((Xs, Ys), dim=-1).reshape(-1, 2)
        derivatives = self.f(0,batch).reshape(
            self.grid_definition[0], self.grid_definition[1], 2
        )
        Us, Vs = derivatives.unbind(dim=-1)

        self.ax.streamplot(
            Xs.numpy().astype(float),
            Ys.numpy().astype(float),
            Us.numpy().astype(float),
            Vs.numpy().astype(float),
            **stream_kwargs,
        )

    def solve_ivp(
        self,
        t_span,
        y_init: torch.tensor = None,
        set_lims=False,
        ivp_kwargs=None,
        plot_kwargs=None,
    ):
        if plot_kwargs is None:
            plot_kwargs = {"color": "red"}
        if ivp_kwargs is None:
            ivp_kwargs = {"solver": "tsit5", 'atol': 1e-9, 'rtol': 1e-12}

        t_eval, trajectories = odeint(self.f, y_init, t_span, **ivp_kwargs)

        self.ax.plot( *trajectories.unbind(dim=-1), **plot_kwargs)

        if set_lims:
            self._plot_vector_field(trajectories)
            plt.autoscale(enable=False)

        return trajectories

    def approx(
        self,
        approx,
        t_init,
        Dapprox=None,
        num_arrows: int = 10,
        **kwargs,
    ):
        if self.t_init != t_init:
            self.t_init = t_init
            self.lines = self.ax.plot(*approx.unbind(-1), **kwargs)
        else:
            self.lines.set_data(*approx.unbind(-1))

    def grab_frame(self):
        self.fig.canvas.draw()
        self.writer.grab_frame()


class LsPlotter:
    def __init__(self, func, Dfunc, plot_res=1000, alpha_max=10):
        self.fig, self.ax = plt.subplots()
        self.phi = func
        self.Dphi = Dfunc
        self.plot_res = plot_res
        self.alpha_max = alpha_max

        (self.alpha_line_art,) = self.ax.plot(
            [], [], color="#EE2233", label="$\phi(a)$"
        )

        (self.alpha_slope,) = self.ax.plot([], [], color="green", label="$\phi'(a)$")

        (self.curpoint,) = self.ax.plot(
            [], [], color="#FF5566", label="$a_i$", marker="o"
        )

        (self.wolfe1_line,) = self.ax.plot(
            [],
            [],
            color="orange",
            label="$\phi(0) + c_1 a \phi'$",
            linestyle="--",
        )

        (self.wolfe2_line,) = self.ax.plot(
            [],
            [],
            color="yellow",
            label="$-c_2 \phi'(0)$",
            linestyle="--",
        )

        (self.ai,) = self.ax.plot([], [], "o", color="#440834")

    def line_search(self, a_cur, c1):
        # plot the line along the search direction
        a = torch.linspace(0, 1.5*a_cur, self.plot_res)
        phi_a = torch.stack([self.phi(a) for a in a])
        self.alpha_line_art.set_xdata(a)
        self.alpha_line_art.set_ydata(phi_a)

        self.alpha_slope.set_xdata([a_cur - 0.1, a_cur + 0.1])
        self.alpha_slope.set_ydata(
            [
                self.phi(a_cur) - 0.1 * self.Dphi(a_cur),
                self.phi(a_cur) + 0.1 * self.Dphi(a_cur),
            ]
        )

        self.ax.set_xlim(-0.1*a_cur, 1.5*a_cur )
        self.ax.set_ylim([0.8 * torch.min(phi_a), self.phi(a_cur *1.5) * 1.5])

        # plot the current point
        self.curpoint.set_xdata([a_cur])
        self.curpoint.set_ydata([self.phi(a_cur)])

        # plot the wolfe1 slope at start point
        self.wolfe1_line.set_xdata([tensor(0), a_cur + 0.5])
        self.wolfe1_line.set_ydata(
            [self.phi(0), self.phi(0) + (a_cur + 0.5) * c1 * self.Dphi(0)]
        )
        # self.wolfe2_line.set_xdata([tensor(0), a_cur+0.5])
        # self.wolfe2_line.set_ydata(
        #     [self.phi(0), self.phi(0) + (a_cur + 0.5) * c1 * self.Dphi(0)]
        # )

    def zoom(self, ai, c1, c2, DPhi_0):
        phi_ai = self.phi(ai)
        Dphi_ai = self.Dphi(ai)
        self.ai.set_xdata([ai])
        self.ai.set_ydata([phi_ai])

        self.alpha_slope.set_xdata([ai, ai + 0.1])
        self.alpha_slope.set_ydata(
            [
                phi_ai,
                phi_ai + 0.1 * abs(Dphi_ai),
            ]
        )

        self.wolfe2_line.set_xdata([ai, ai + 0.1])
        self.wolfe2_line.set_ydata([phi_ai, phi_ai - 0.1 * c2 * DPhi_0])

        wait()

    def close_all(self):
        plt.close(self.fig)
