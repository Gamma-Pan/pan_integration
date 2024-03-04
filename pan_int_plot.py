import torch
from torch import tensor, sin, cos, tanh
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
import matplotlib.pyplot as plt

from torch import pi as PI

def spiral(batch, a=0.2):
    P = tensor([[-a, 1.0], [-1.0, -a]])[None, None]

    xy = batch[..., None]
    derivative = (P @ xy)
    return torch.squeeze(derivative)


def voltera_lotka(batch):
    x = batch[...,0]
    y = batch[...,1]
    a = 0.4
    b = 0.9
    c = 1.3
    d = 0.2

    derivative = torch.stack( (a*x - b*x*y ,d*x*y -c*y ) )
    return torch.squeeze(derivative)

if __name__ == "__main__":
    global COUNTER

    NUM_POINTS = 30
    f = spiral

    y_init = tensor([-1.2, -3.4], dtype=torch.float)
    ax_kwargs = {"xlim": (-5, 5), "ylim": (-5, 5)}
    plotter = plotting.VfPlotter(
        f,
        y_init,
        t_init=0.,
        ax_kwargs=ax_kwargs,
    )

    # plotter.solve_ivp(
    #     [0, 20], ivp_kwargs={"method": "BDF"}, plot_kwargs={"color": "red", 'label': 'BDF'}
    # )
    # plotter.solve_ivp(
    #     [0, 20], ivp_kwargs={"method": "RK45"}, plot_kwargs={"color": "brown", 'label': 'RK45'}
    # )
    # plotter.solve_ivp(
    #     [0, 3], ivp_kwargs={"method": "RK23"}, plot_kwargs={"color": "lime", 'label': 'RK23'}
    # )
    # plotter.solve_ivp(
    #     [0, 3], ivp_kwargs={"method": "Radau"}, plot_kwargs={"color": "salmon", 'label': 'Radau'}
    # )
    plotter.solve_ivp(
        [0, 20], ivp_kwargs={"method": "LSODA"}, plot_kwargs={"color": "salmon", 'label': 'LSODA'}
    )


    @torch.no_grad()
    def callback(B_vec, y_init, cur_interval, num_coeff_per_dim, dims):
        t_0 = torch.tensor(cur_interval[0])

        t_points = torch.linspace(cur_interval[0], cur_interval[1], NUM_POINTS)[:, None]
        freqs = torch.arange(num_coeff_per_dim // 2, dtype=torch.float)[None]
        grid = (t_points * freqs)
        Phi_cT = torch.cos(grid)
        Phi_sT = torch.sin(grid)
        D_Phi_cT = -freqs * sin(grid)
        D_Phi_sT = freqs * cos(grid)

        #  the first half of B comprises the Bc matrix minus the first row
        Bc_m1 = B_vec[:(num_coeff_per_dim // 2 - 1) * dims].reshape(num_coeff_per_dim // 2 - 1, dims)
        # the rest comprises the Bs matrix minus the first two rows
        Bs_m2 = B_vec[(num_coeff_per_dim // 2 - 1) * dims:].reshape(num_coeff_per_dim // 2 - 2, dims)

        # calculate the 1-index row of Bs
        Bs_1 = ((f(y_init[None])
                 + sin(t_0) * Bc_m1[0:1, :]
                 - D_Phi_cT[0, 2:] @ Bc_m1[1:, :]
                 - D_Phi_sT[0, 2:] @ Bs_m2)
                / cos(t_0))

        # calculate the 1-index row of Bs
        Bs_m1 = torch.vstack((Bs_1, Bs_m2))

        # calculate the first row of Bc
        Bc_0 = y_init[None] - Phi_cT[0:1, 1:] @ Bc_m1 - Phi_sT[0:1, 1:] @ Bs_m1
        Bc = torch.vstack((Bc_0, Bc_m1))

        # the first row Bs doesn't contribute
        Bs = torch.vstack((torch.zeros(1, dims), Bs_m1))

        # FOR PLOTTING
        t_points = torch.linspace(*cur_interval, 100)[:, None]
        freqs = torch.arange(num_coeff_per_dim // 2, dtype=torch.float)[None]
        grid = t_points * freqs
        Phi_cT = torch.cos(grid)
        Phi_sT = torch.sin(grid)

        approx = Phi_cT @ Bc + Phi_sT @ Bs
        Dapprox = D_Phi_cT @ Bc + D_Phi_sT @ Bs

        plotter.pol_approx(approx, t_0)
        plotting.wait()


    approx = pan_int(
        spiral,
        y_init,
        callback=callback,
        t_lims=[0, 20],
        step=5.,
        num_points=NUM_POINTS,
        num_coeff_per_dim=20,
        etol=1e-3
    )

    print("FIN")
    plt.show()
