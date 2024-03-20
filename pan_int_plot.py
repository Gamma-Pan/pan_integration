import torch
from torch import tensor, sin, cos, tanh, ones, arange, sigmoid
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
from pan_integration.solvers.pan_integration import _B_init_cond, _cheb_phis
import matplotlib as mpl

def spiral(batch):
    a = 0.5
    P = tensor([[-a, 1.0], [-1.0, -a]])[None]
    xy = batch[..., None]  # add trailing dim for matrix vector muls
    # derivative = tanh(-P @ sin(0.5 * P @ tanh(P @ xy)))
    # derivative = P @ xy
    derivative = tanh(P @ xy)
    return torch.squeeze(derivative)


if __name__ == "__main__":
    global COUNTER

    t_lims = [0, 6]
    y_init = tensor([-1, -1], dtype=torch.float)
    f = spiral

    ax_kwargs = {"xlim": (-1.4, 0.8), "ylim": (-1.3, 0.9)}
    plotter = plotting.VfPlotter(
        f,
        y_init,
        t_init=0.0,
        ax_kwargs=ax_kwargs,
        animation=False,
        queue_size=5,
        text=False,
    )

    # plotter.solve_ivp(
    #     t_lims,
    #     ivp_kwargs={
    #         "method": "RK45",
    #         "max_step": 1e-3,
    #     },  # , "atol": 1e-8, "rtol": 1e-13},
    #     plot_kwargs={"color": "lime", "alpha": 0.7},
    # )
    # plotter.solve_ivp(
    #     t_lims,
    #     ivp_kwargs={"method": "DOP853", "atol": 1e-8, "rtol": 1e-13},
    #     plot_kwargs={"color": "turquoise", "alpha": 0.7},
    # )
    # plotter.solve_ivp(
    #     t_lims,
    #     ivp_kwargs={"method": "BDF", "atol": 1e-8, "rtol": 1e-13},
    #     plot_kwargs={"color": "firebrick", "alpha": 0.7},
    # )
    # plotter.solve_ivp(
    #     t_lims,
    #     ivp_kwargs={"method": "LSODA", "max_step": 1e-3},
    #     plot_kwargs={"color": "violet", "alpha": 1},
    # )

    # plot the legend
    # plotter.ax.legend(loc="lower right")

    @torch.no_grad()
    def callback(B_vec, y_init, cur_interval):
        t_0 = cur_interval[0]
        dims = y_init.shape[0]
        num_coeff_per_dim = (B_vec.shape[0] + 2 * dims) // dims

        # plotting points
        Phi, DPhi = _cheb_phis(1000, num_coeff_per_dim, cur_interval)
        B = _B_init_cond(
            B_vec.reshape(dims, num_coeff_per_dim - 2).T, y_init, f(y_init), Phi, DPhi
        )

        approx = Phi @ B
        Dapprox = DPhi @ B

        plotter.approx(approx, t_0, Dapprox=None, color=(0.1, 0.4, 0.12))
        # plotting.wait()
        plotter.fig.canvas.draw()
        plotter.fig.canvas.flush_events()
        if plotter.animation:
            plotter.grab_frame()

    approx, nfe = pan_int(
        spiral,
        y_init,
        callback=callback,
        t_lims=t_lims,
        step=(t_lims[1] - t_lims[0]) / 1,
        num_points=100,
        num_coeff_per_dim=25,
        atol=1e3,
        etol=5e-1,
        plotter=plotter,
        coarse_steps=7,
    )

    print(
        f"pan int \t took {nfe} function evaluations y(T) = ({approx[-1, 0].item():.6} , {approx[-1, 1].item():.6})"
    )
    if plotter.animation:
        plotter.writer.finish()

    plotter.ax.plot(approx[-1, 0], approx[-1, 1], "o", color="blue", zorder=100)
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
    plotter.ax.set_xlabel('$y_0$')
    plotter.ax.set_ylabel('$y_1$')
    plotter.fig.savefig("./latex/sol_regression.pgf")
    mpl.pyplot.show()

    print("FIN")
