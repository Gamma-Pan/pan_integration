import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
from itertools import cycle

from matplotlib.animation import PillowWriter
from scipy.integrate import solve_ivp
import torch

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
        y_init: torch.Tensor,
        t_init: float,
        grid_definition: tuple = (40, 40),
        existing_axes: plt.Axes = None,
        ax_kwargs: dict = None,
        show=False,
        animation=False,
        text=False,
        queue_size=5,
    ):
        self.y_init = y_init
        self.t_init = t_init

        if existing_axes is None:
            self.fig, self.ax = plt.subplots()
            if ax_kwargs is not None:
                self.ax.set(**ax_kwargs)
        else:
            self.ax = existing_axes

        self.ax.plot(*y_init, "ro")
        self.approx_arts = None
        self.queue_size = queue_size
        self.Dapprox_art = None
        self.text = text

        self.grid_definition = grid_definition
        self.f = f

        self._plot_vector_field()

        if animation:
            self.writer = PillowWriter(fps=4, metadata={"artist": "Yiorgos Pan"})
            self.writer.setup(self.fig, "test.gif", 100)
            self.writer.frame_format = "png"
        self.animation = animation

        if show:
            wait()

    def _plot_vector_field(self):
        xs = torch.linspace(*self.ax.get_xlim(), self.grid_definition[0])
        ys = torch.linspace(*self.ax.get_ylim(), self.grid_definition[1])

        Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")

        batch = torch.stack((Xs, Ys), dim=-1).reshape(-1, 2)
        derivatives = self.f(batch).reshape(
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

    def update(self):
        raise NotImplementedError

    def solve_ivp(
        self, interval, y_init: torch.tensor = None, ivp_kwargs=None, plot_kwargs=None
    ):
        plot_kwargs = plot_kwargs or {}
        ivp_kwargs = ivp_kwargs or {}
        y_init = self.y_init if y_init is None else y_init

        ivp_sol = solve_ivp(
            lambda t, x: self.f(torch.tensor(x[None], dtype=torch.float32))
            .squeeze()
            .numpy(),
            interval,
            y_init,
            **ivp_kwargs,
        )

        trajectory = ivp_sol.y
        print(
            f"Method {ivp_kwargs['method']} \t took {ivp_sol.nfev} function evaluations, y(T) = ({trajectory[0][-1]:.6},{trajectory[1][-1]:.6} )"
        )

        (self.trajectory,) = self.ax.plot(
            *trajectory,
            **plot_kwargs,
            label=f"{ivp_kwargs['method']} - {ivp_sol.nfev} NFE",
            zorder=10,
        )
        self.ax.plot(*trajectory[:, -1], "o", color=plot_kwargs["color"], zorder=15)
        return trajectory

    def approx(
        self,
        approx,
        t_init,
        color,
        Dapprox=None,
        num_arrows: int = 10,
        **kwargs,
    ):
        kwargs["color"] = color
        # faster than deleting and creating artists
        if not self.approx_arts:
            self.approx_arts = cycle(
                [
                    (
                        self.ax.text(0, 0, ""),
                        self.ax.plot([], [], zorder=100, **kwargs)[0],
                    )
                    for _ in range(self.queue_size)
                ]
            )
            self.approx_art = next(self.approx_arts)

        # if new step erase previous artists in list expect last and draw new starting point
        if self.t_init != t_init:
            self.t_init = t_init
            self.ax.plot(*approx[0, :], "o", **kwargs)
            for _ in range(self.queue_size):
                art = next(self.approx_arts)
                art.approx_arts[0].set_data([], [])
                art.approx_arts[1].set_text("")

        # update one artist and reduce opacity of the previous ones
        self.approx_art[1].set_data(approx[:, 0], approx[:, 1])
        self.approx_art[1].set_linestyle("-")
        self.approx_art[1].set_linewidth(2)
        self.approx_art[1].set_alpha(1)
        self.approx_art[1].set_c(color)
        if self.text:
            self.approx_art[0].set_position(approx[-1, :])
            self.approx_art[0].set_text(",".join((f"{x:.2f}") for x in approx[-1, :]))
            self.approx_art[0].set_c(color)

        if self.queue_size > 1:
            for a in torch.linspace(1, 0.0, self.queue_size + 1)[1:-1]:
                art = next(self.approx_arts)
                art[0].set_alpha(float(a))
                art[1].set_alpha(float(a))
                art[1].set_linestyle("--")
                art[1].set_linewidth(1)
                new_c = list(mpl.colors.to_rgb(art[0].get_c()))
                new_c[0] = min(1, new_c[0] + 0.05)
                new_c[1] = max(0, new_c[1] - 0.05)
                art[0].set_c(tuple(new_c))
                art[1].set_c(tuple(new_c))
            self.approx_art = art

        if Dapprox is not None:
            sz = approx.shape[0]
            idxs = torch.arange(0, sz, sz // num_arrows).tolist()

            if self.Dapprox_art is not None:
                self.Dapprox_art.remove()

            self.Dapprox_art = self.ax.quiver(
                approx[idxs, 0],
                approx[idxs, 1],
                Dapprox[idxs, 0],
                Dapprox[idxs, 1],
                angles="xy",
            )

    def grab_frame(self):
        self.fig.canvas.draw()
        self.writer.grab_frame()


class LsPlotter:
    def __init__(self, func, plot_res=100, alpha_max=10):
        self.fig, self.ax = plt.subplots()
        self.phi = func
        self.plot_res = plot_res
        self.alpha_max = alpha_max

        (self.alpha_line_art,) = self.ax.plot(
            [], [], color="#EE2233", label="$\phi(a)$"
        )
        (self.curpoint,) = self.ax.plot(
            [], [], color="#FF5566", label="$a_i$", marker="o"
        )

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
