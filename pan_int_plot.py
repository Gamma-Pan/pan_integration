import torch
from torch import tensor, sin, cos, tanh, ones, arange, sigmoid
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
from pan_integration.solvers.pan_integration import _phis_init_cond
import matplotlib.pyplot as plt
from torch import pi as PI


def spiral(batch):
    a = 1
    P = tensor([[-a, 1.0], [-1.0, -a]])[None, None]
    xy = batch[..., None]
    derivative = tanh(-P @ sin(0.5 * P @ tanh(P @ xy))) + 0.1
    # derivative = P@xy+a
    return torch.squeeze(derivative)


if __name__ == "__main__":
    global COUNTER

    t_lims = [0, 5]
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

    plotter.solve_ivp(t_lims, ivp_kwargs={"method": "LSODA", "atol": 1e-9, "max_step":1e-2},
                      plot_kwargs={"color": "brown", "alpha": 0.8},)
    plotter.solve_ivp(t_lims, ivp_kwargs={"method": "RK45" },
                      plot_kwargs={"color": "lime", "alpha": 0.5})
    # plotter.solve_ivp(t_lims, ivp_kwargs={"method": "DOP853", "atol": 1e-6},
    #                   plot_kwargs={"color": "turquoise", 'alpha': 0.5})

    # plot the legend
    plotter.ax.legend(loc='upper right')


    @torch.no_grad()
    def callback(B_vec, y_init, cur_interval):
        t_0 = cur_interval[0]
        step = cur_interval[1] - cur_interval[0]
        dims = y_init.shape[0]
        num_coeff_per_dim = (B_vec.shape[0] + 2 * dims) // dims

        # CONVERT VECTOR BACK TO MATRICES
        #  the first half of B comprises the Bc matrix minus the first row
        Bc = B_vec[:(num_coeff_per_dim // 2 - 1) * dims].reshape(num_coeff_per_dim // 2 - 1, dims)
        # the rest comprises the Bs matrix minus the first row
        Bs = B_vec[(num_coeff_per_dim // 2 - 1) * dims:].reshape(num_coeff_per_dim // 2 - 1, dims)

        # FOR PLOTTING
        Phi_c , Phi_s,  Phi_s_1, _ ,_ ,_ = _phis_init_cond(step, 300, num_coeff_per_dim)

        approx = Phi_c @ Bc + Phi_s @ Bs + Phi_s_1 @ f(y_init)[None] + y_init

        plotter.approx(approx, t_0, alpha=0.7)
        # plotting.wait()
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
        num_points=30,
        num_coeff_per_dim=5,
        etol=1e-9,
        plotter=plotter
    )

    print(f"pan solution : ({approx[-1, 0].item():.6} , {approx[-1, 1].item():.6})")
    if plotter.animation:
        plotter.writer.finish()

    plotter.ax.plot(approx[-1, 0], approx[-1, 1], 'o', color='green' )
    plotting.wait()
    print("FIN")
