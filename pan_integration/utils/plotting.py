import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
from itertools import cycle

from matplotlib.animation import PillowWriter
import torch
from torch import tensor, abs
from ..core.solvers import T_grid, DT_grid

from torchdyn.numerics import odeint


mpl.use("TkAgg")

quiver_args = {
    "headwidth": 2,
    "headlength": 2,
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
        grid_definition: int = 40,
        existing_axes: plt.Axes = None,
        ax_kwargs: dict = None,
        animation=False,
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
        self.grid_definition: int = grid_definition
        self.f = f

        if animation:
            self.writer = PillowWriter(fps=4, metadata={"artist": "Yiorgos Pan"})
            self.writer.setup(self.fig, "test.gif", 100)
            self.writer.frame_format = "png"
        self.animation = animation

    def _plot_vector_field(
        self, trajectories, xmin=None, xmax=None, ymin=None, ymax=None
    ):
        if trajectories is not None:
            padding = 0.1
            xmax = float(torch.max(trajectories) + padding)
            xmin = float(torch.min(trajectories) - padding)
            ymax = float(torch.max(trajectories) + padding)
            ymin = float(torch.min(trajectories) - padding)


        # win_sz = max(xmax - xmin, ymax - ymin) / 2
        #
        # xs = torch.linspace(
        #     (xcenter := (xmin + xmax) / 2) - win_sz,
        #     xcenter + win_sz,
        #     self.grid_definition,
        # ).to(self.device)
        # ys = torch.linspace(
        #     (ycenter := (ymin + ymax) / 2) - win_sz,
        #     ycenter + win_sz,
        #     self.grid_definition,
        # ).to(self.device)

        win_sz = max(xmax - xmin, ymax - ymin) / 2

        xs = torch.linspace(
            xmin,
            xmax,
            self.grid_definition,
        ).to(self.device)
        ys = torch.linspace(
            xmin,
            xmax,
            self.grid_definition,
        ).to(self.device)

        # self.ax.set_xlim(xcenter - win_sz, xcenter+win_sz)
        # self.ax.set_ylim(ycenter - win_sz, ycenter+win_sz)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin ,ymax)

        Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")

        batch = torch.stack((Xs, Ys), dim=-1).reshape(-1, 2)
        derivatives = self.f(0, batch).reshape(
            self.grid_definition, self.grid_definition, 2
        )
        Us, Vs = derivatives.unbind(dim=-1)

        self.ax.streamplot(
            Xs.cpu().numpy().astype(float),
            Ys.cpu().numpy().astype(float),
            Us.cpu().numpy().astype(float),
            Vs.cpu().numpy().astype(float),
            **stream_kwargs,
        )

    def solve_ivp(
        self,
        t_span,
        y_init: torch.tensor = None,
        set_lims=False,
        end_point = True,
        ivp_kwargs=None,
        plot_kwargs=None,
    ):
        if plot_kwargs is None:
            plot_kwargs = {"color": "red"}
        if ivp_kwargs is None:
            ivp_kwargs = {"solver": "tsit5", "atol": 1e-9, "rtol": 1e-12}

        self.device = y_init.device
        t_eval, trajectories = odeint(self.f, y_init, t_span, **ivp_kwargs)
        trajectories = trajectories.cpu()

        self.ax.plot(*trajectories.unbind(dim=-1), **plot_kwargs)

        if end_point:
            self.ax.scatter(*trajectories[-1].unbind(dim=-1), marker='o' ,color=plot_kwargs["color"] )

        if set_lims:
            self._plot_vector_field(trajectories)
            plt.autoscale(enable=False)

        return trajectories

    @staticmethod
    def _approx_from_B(B, t_lims, num_points=100):
        dims = len(B.shape) - 1
        num_coeff = B.shape[-1]
        t = torch.linspace(-1, 1, num_points)
        Phi = T_grid(t, num_coeff).to(B.device)
        DPhi = (
            2 / (t_lims[1].cpu() - t_lims[0].cpu()) * DT_grid(t, num_coeff).to(B.device)
        )
        approx = B @ Phi
        Dapprox = B @ DPhi
        return (
            approx.permute(-1, *list(range(dims))),
            Dapprox.permute(-1, *list(range(dims))),
        )

    def approx(
        self,
        t_lims,
        B=None,
        show_arrows=False,
        num_arrows: int = 10,
        num_points=100,
        **kwargs,
    ):
        t_init = t_lims[0]
        approx, Dapprox = self._approx_from_B(B, t_lims, num_points)

        every_num_points = num_points // num_arrows

        # approx = approx + y_init
        if self.t_init != t_init:
            self.t_init = t_init
            self.lines = self.ax.plot(*approx.cpu().unbind(-1), **kwargs)
            if show_arrows:
                self.arrows = self.ax.quiver(
                    *approx[::every_num_points].cpu().unbind(-1), *Dapprox[::every_num_points].cpu().unbind(-1), **quiver_args
                )
                self.farrows = self.ax.quiver(
                    *approx[::every_num_points].cpu().unbind(-1),
                    *self.f(0, approx)[::every_num_points].cpu().unbind(-1),
                    **quiver_args,
                    color="red",
                )
        else:
            for line, data in zip(self.lines, approx.cpu().unbind(-2)):
                line.set_data(*data.unbind(-1))

            if show_arrows:
                self.arrows.set_UVC(*Dapprox[::every_num_points].cpu().unbind(-1))
                self.farrows.set_UVC(*self.f(0, approx[::every_num_points]).cpu().unbind(-1))

                self.arrows.set_offsets(approx[::every_num_points].cpu().reshape(-1, 2))
                self.farrows.set_offsets(approx[::every_num_points].cpu().reshape(-1, 2))

        return approx

    def grab_frame(self):
        self.fig.canvas.draw()
        self.writer.grab_frame()

    def wait(self):
        wait()


class LsPlotter:
    def __init__(self, func, Dfunc, plot_res=10_000, alpha_max=10):
        self.fig, self.ax = plt.subplots()
        self.phi = func
        self.Dphi = Dfunc
        self.plot_res = plot_res
        self.alpha_max = alpha_max

        (self.alpha_line_art,) = self.ax.plot(
            [], [], color="#EE2233", label="$\\phi(a)$"
        )

        (self.alpha_slope,) = self.ax.plot([], [], color="green", label="$\\phi'(a)$")

        (self.curpoint,) = self.ax.plot(
            [], [], color="#FF5566", label="$a_i$", marker="o"
        )

        (self.wolfe1_line,) = self.ax.plot(
            [],
            [],
            color="orange",
            label="$\\phi(0) + c_1 a \\phi'$",
            linestyle="--",
        )

        (self.wolfe2_line,) = self.ax.plot(
            [],
            [],
            color="yellow",
            label="$-c_2 \\phi'(0)$",
            linestyle="--",
        )

        (self.ai,) = self.ax.plot([], [], "o", color="#440834")

    def line_search(self, a_cur, c1):
        # plot the line along the search direction
        a = torch.linspace(0, 1.5 * a_cur, self.plot_res)
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

        self.ax.set_xlim(-0.1 * a_cur, 1.5 * a_cur)
        self.ax.set_ylim([0.8 * torch.min(phi_a), self.phi(a_cur * 1.5) * 1.5])

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


# class DimPlotter():
#
#     def callback(B, t_lims, y_init):
#         dims = len(B.shape) - 1
#         num_coeff = B.shape[-1]
#         t = torch.linspace(-1, 1, 100).to(device)
#         Phi = T_grid(t, num_coeff)
#         DPhi = 2 / (t_lims[1] - t_lims[0]) * DT_grid(t, num_coeff)
#
#         approx = B @ Phi
#         Dapprox = B @ DPhi
#
#         t_sol = torch.linspace(*t_lims, 100)
#         _, sol = torchdyn.numerics.odeint(
#             vf, solver="tsit5", t_span=t_sol, save_at=t_sol, x=y_init
#         )
#         batch = torch.randint(0,BATCH_SIZE-1,(1,))
#         channel = torch.randint(0, 8-1, (1,))
#         dim1= torch.randint(0,5, (1,))
#         dim2 = torch.randint(0,5, (1,))
#         yy_init = y_init[batch, channel, dim1, dim2].cpu()
#         ax.set_ylim( yy_init -1, yy_init+1)
#         line_sol.set_ydata(sol[:, batch, channel, dim1, dim2].cpu())
#         line_approx.set_ydata(approx[batch, channel, dim1, dim2, :].mT.cpu())
#
#         fig.canvas.flush_events()
#         fig.canvas.draw()
