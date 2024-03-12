import torch
from torch import tensor, sin, cos, tanh, ones, arange, sigmoid
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
from pan_integration.solvers.pan_integration import _B_init_cond, _cheb_phis
import matplotlib.pyplot as plt
from torch import pi as PI


def spiral(batch):
    a = 0.3
    P = tensor([[-a, 1.0], [-1.0, -a]])[None, None]
    xy = batch[..., None]
    derivative = torch.exp(-P @ sin(0.5 * P @ tanh(P @ xy)))
    # derivative = P@xy+a
    return torch.squeeze(derivative)


if __name__ == "__main__":
    global COUNTER

    t_lims = [0, 10]
    y_init = tensor([-2, -2], dtype=torch.float)
    f = spiral

    ax_kwargs = {"xlim": (-3, 3), "ylim": (-3, 3)}
    plotter = plotting.VfPlotter(
        f,
        y_init,
        t_init=0.,
        ax_kwargs=ax_kwargs,
        animation=True
    )

    plotter.solve_ivp(t_lims, ivp_kwargs={"method": "LSODA", "atol": 1e-9, "max_step": 1e-2},
                      plot_kwargs={"color": "brown", "alpha": 0.8}, )
    plotter.solve_ivp(t_lims, ivp_kwargs={"method": "RK45", "atol": 1e-9 },
                      plot_kwargs={"color": "lime", "alpha": 0.5})
    # plotter.solve_ivp(t_lims, ivp_kwargs={"method": "DOP853", "atol": 1e-6},
    #                   plot_kwargs={"color": "turquoise", 'alpha': 0.5})

    # plot the legend
    plotter.ax.legend(loc='upper right')


    @torch.no_grad()
    def callback(B_vec, y_init, cur_interval):
        t_0 = cur_interval[0]
        dims = y_init.shape[0]
        num_coeff_per_dim = (B_vec.shape[0] + 2 * dims) // dims

        # plotting points
        Phi,DPhi = _cheb_phis(100, num_coeff_per_dim, cur_interval)
        B = _B_init_cond(B_vec.reshape(dims, num_coeff_per_dim - 2).T, y_init, f(y_init),  Phi, DPhi)

        approx = Phi @ B
        Dapprox = DPhi @ B

        plotter.approx(approx, t_0, Dapprox=Dapprox, alpha=0.7)
        plotting.wait()
        plotter.fig.canvas.draw()
        plotter.fig.canvas.flush_events()

        if plotter.animation:
            plotter.grab_frame()


    approx = pan_int(
        spiral,
        y_init,
        callback=callback,
        t_lims=t_lims,
        step=None,
        num_points=100,
        num_coeff_per_dim=30,
        etol=1e-9,
        plotter=plotter
    )

    print(f"pan solution : ({approx[-1, 0].item():.6} , {approx[-1, 1].item():.6})")
    if plotter.animation:
        plotter.writer.finish()

    plotter.ax.plot(approx[-1, 0], approx[-1, 1], 'o', color='green')
    plotting.wait()
    print("FIN")
