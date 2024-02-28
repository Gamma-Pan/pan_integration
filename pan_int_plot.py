import torch
from torch import tensor
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
import matplotlib.pyplot as plt


def spiral(batch, a=0.2):
    P = tensor([[-a, 1.0], [-1.0, -a]])[None, None]

    xy = batch[..., None]
    derivative = P @ xy
    return torch.squeeze(derivative)


if __name__ == "__main__":
    y_init = tensor([-3, -3], dtype=torch.float)
    ax_kwargs = {"xlim": (-5, 5), "ylim": (-5, 5)}
    plotter = plotting.VfPlotter(
        spiral,
        y_init,
        ax_kwargs=ax_kwargs,
    )

    plotter.solve_ivp(
        [0, 2], ivp_kwargs={"method": "LSODA"}, plot_kwargs={"color": "red"}
    )


    @torch.no_grad()
    def callback(B, Phi_cT, Phi_sT, y_init, cur_interval, num_coeff_per_dim, dims):
        B_size = B.shape[0]
        B_c = B[: B_size // 2].reshape(num_coeff_per_dim // 2 - 1, dims)
        B_s = B[B_size // 2:].reshape(num_coeff_per_dim // 2 - 1, dims)

        # calculate the first row of B_c
        B_c0 = y_init[None] - Phi_cT[0:1, 1:] @ B_c - Phi_sT[0:1, 1:] @ B_s
        B_c = torch.vstack((B_c0, B_c))

        # the first row B_s doesn't contribute
        B_s = torch.vstack((torch.zeros(1, dims), B_s))
        t_points = torch.linspace(*cur_interval, 100)[:, None]
        freqs = torch.arange(num_coeff_per_dim//2, dtype=torch.float)[None]
        grid = t_points * freqs
        Phi_cT = torch.cos(grid)
        Phi_sT = torch.sin(grid)

        approx = Phi_cT @ B_c + Phi_sT @ B_s

        plotter.pol_approx(approx)
        plotting.wait()


    approx = pan_int(
        spiral,
        y_init,
        callback=callback,
        t_lims=[0, 2],
        num_points=10,
        num_coeff_per_dim=4,
    )

    plotter.pol_approx(approx, )
    print("FIN")
    plt.show()
