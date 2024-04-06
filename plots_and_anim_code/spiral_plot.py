import torch
from torch import tensor, tanh, cosh, sin
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
from pan_integration.solvers.pan_integration import (
    _B_init_cond,
    _cheb_phis,
    _coarse_euler_init,
)
import matplotlib as mpl
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def spiral(batch):
    a = 0.5
    P = tensor([[-a, 1.0], [-1.0, -a]], dtype=torch.float)[None]
    xy = batch[..., None].to(torch.float32)  # add trailing dim for matrix vector muls
    derivative = P @ xy
    return torch.squeeze(derivative).to(torch.float64)


def regression(func, t_lims, y_init, save=False):
    ax_kwargs = {"xlim": (-1.5, 0.5), "ylim": (-1.2, 0.8)}

    plotter = plotting.VfPlotter(
        func,
        y_init,
        t_init=0.0,
        ax_kwargs=ax_kwargs,
        animation=False,
        queue_size=3,
        text=False,
    )

    trajectory = plotter.solve_ivp(
        t_lims,
        ivp_kwargs={
            "method": "RK45",
            "max_step": 1e-2,
        },
        plot_kwargs={"color": "firebrick", "alpha": 0.5},
        set_lims=True,
    )

    y_TRUE = trajectory[-1, :]

    methods = ["RK45"]  # ["RK45", "DOP853", "BDF", "LDSODA"]
    colors = ["lime"]  # ["lime", "turquoise", "firebrick", "violet"]
    for method, color in zip(methods, colors):
        plotter.solve_ivp(
            t_lims,
            ivp_kwargs={
                "method": method,
                "atol": 1e-9,
                "rtol": 1e-13,
            },
            plot_kwargs={"color": color, "alpha": 1},
        )

    @torch.no_grad()
    def callback(B_vec, y_init, cur_interval):
        t_0 = cur_interval[0]
        dims = y_init.shape[0]
        num_coeff_per_dim = (B_vec.shape[0] + 2 * dims) // dims

        # plotting points
        Phi, DPhi = _cheb_phis(100, num_coeff_per_dim, cur_interval)
        B = _B_init_cond(
            B_vec.reshape(dims, num_coeff_per_dim - 2).T,
            y_init,
            func(y_init),
            Phi,
            DPhi,
        )

        approx = Phi @ B
        Dapprox = DPhi @ B

        plotter.approx(approx, t_0, Dapprox=None, color=(0.1, 0.4, 0.12))
        # plotting.wait()
        # plt.pause(0.3)
        plotter.fig.canvas.draw()
        plotter.fig.canvas.flush_events()
        if plotter.animation:
            plotter.grab_frame()

    approx, (ls_nfe, nfe) = pan_int(
        lambda x: func(x.to(torch.float32)).to(torch.get_default_dtype()),
        y_init,
        callback=None,
        t_lims=t_lims,
        step=(t_lims[1] - t_lims[0])/4,
        num_points=300,
        num_coeff_per_dim=300,
        etol_newton=1e-14,
        etol_lstsq=1e-8,
        coarse_steps=5,
    )

    print(
        f"Method pan took {ls_nfe,nfe}\t\t function evaluations, y(T) = ({approx[-1, 0].item():.9f},{approx[-1, 1].item():.9f}), "
    )

    plotter.ax.plot(approx[-1, 0], approx[-1, 1], "o", color="blue", zorder=100)

    if save:
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
        plotter.ax.set_xlabel("$y_0$")
        plotter.ax.set_ylabel("$y_1$")
        plotter.fig.savefig("./latex/ivps.pgf")

    mpl.pyplot.show()


if __name__ == "__main__":
    t_lims = tensor([0, 8])
    y_init = tensor([0.5, -5.0])
    regression(spiral, t_lims, y_init)
