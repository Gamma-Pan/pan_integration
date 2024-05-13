import torch
from torch import tensor, cos, sin, tanh, log, cosh
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import PillowWriter, FFMpegWriter
from scipy.integrate import solve_ivp
from math import floor, log10

from typing import List

# mpl.use('TkAgg')
from pan_integration.numerics import pan_int
from pan_integration.numerics.functional import _cheb_phis, _B_init_cond
from pan_integration.utils import plotting
from torch import pi as PI

torch.manual_seed(434)


def f(x):
    return sin(2 * PI * (log(10 * x**2 + 3 * x + 4)))


def f2(x):
    a = 0.4
    P = tensor([[-a, 1.0], [-1.0, -a]])[None]
    return tanh(P @ x[..., None]).squeeze()


def lorenz(batch, sigma=10, rho=28, beta=8 / 3):
    x, y, z = batch.unbind(dim=-1)
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return torch.stack([dx, dy, dz], dim=-1).squeeze()


def lorenz_anim(t_lims, y_init):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sol = solve_ivp(
        lambda t, y: lorenz(tensor(y, dtype=torch.float32)[None]),
        t_lims,
        y_init,
        atol=1e-6,
        rtol=1e-6,
        # max_step=1e-2,
    )
    print(sol.nfev)
    print(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1])
    (solax,) = ax.plot(sol.y[0], sol.y[1], sol.y[2], alpha=0.5)
    solax.remove()
    plt.autoscale(False)

    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    (approx_art,) = ax.plot([], [], [], color="#125577", alpha=0.7)
    plt.pause(0.05)

    writer = FFMpegWriter(fps=24, metadata={"artist": "Yiorgos Pan"})
    writer.setup(fig, "test.mp4", 100)
    # writer.frame_format = "png"
    azim = 0

    curt = t_lims[0]

    def callback(B_vec, y_init, cur_interval):
        nonlocal azim
        azim += 0.2
        t_0 = cur_interval[0]
        dims = y_init.shape[0]
        num_coeff_per_dim = (B_vec.shape[0] + 2 * dims) // dims

        # plotting points
        Phi, DPhi = _cheb_phis(100, num_coeff_per_dim, cur_interval, device=y_init.device)
        B = _B_init_cond(
            B_vec.reshape(dims, num_coeff_per_dim - 2).T,
            y_init,
            lorenz(y_init),
            Phi,
            DPhi,
        )

        nonlocal approx_art
        nonlocal curt
        approx = Phi @ B
        if torch.isclose(tensor(curt), tensor(cur_interval[0])):
            approx_art.set_data_3d(*approx.unbind(dim=-1))
        else:
            curt = cur_interval[0]
            (approx_art,) = ax.plot(*approx.unbind(dim=-1), color="#125577", alpha=0.7)
        ax.view_init(azim=azim)
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()

    approx, (nfe_ls, nfe_hess) = pan_int(
        lorenz,
        y_init,
        t_lims,
        50,
        100,
        0.3,
        ls_kwargs=dict(etol=1e-8),
        newton_kwargs=dict(etol=1e-8),
        callback=callback,
    )
    writer.finish()
    plt.close("all")
    print(nfe_ls, nfe_hess)
    print(approx[-1, :])


def latex_table(
    f,
    y_init,
    t_lims,
    scipy_params: list,
    pan_params: list,
    filename="test.txt",
):
    ff = lambda t, x: f(tensor(x)[None]).numpy()

    file = open(filename, "w")

    head = """
    \\begin{center}
        \\renewcommand{\\arraystretch}{1.5}
        \\renewcommand{\\tabcolsep}{10.25pt}
        \\begin{tabular}{|c|c|c|c|}
            \hline
            Method(params) & tolerance & error(t) & NFE \\\\
            \hline
    """
    file.write(head)
    file.close()

    file = open(filename, "a")

    # true
    true_sol = solve_ivp(
        ff,
        t_lims,
        y_init,
        method="RK45",
        max_step=1e-2,
    )

    true_y = true_sol.y[:, -1]
    for param in scipy_params:
        sol = solve_ivp(
            ff,
            t_lims,
            y_init,
            method=param[0],
            atol=param[1],
            rtol=1e-14,
        )

        error = torch.norm(tensor(true_y - sol.y[:, -1])).numpy()
        nfe = sol.nfev
        print(sol.nfev)

        row = f"\t\t\t {param[0]} & {param[1]:1.1e} & {error:1.2e} & {nfe}  \n \t\t\t \\\\ \hline \n"
        file.write(row)

    for param in pan_params:
        sol, (ls_nfe, newt_nfe) = pan_int(
            f,
            y_init,
            t_lims,
            num_coeff_per_dim=param[0],
            num_points=300,
            step=param[1],
            etol_lstsq=param[2],
            etol_newton=param[3],
        )
        print(ls_nfe, newt_nfe)

        error = torch.norm(tensor(true_y) - sol[-1, :]).numpy()
        row = f"\t\t\t PAN({param[0]}) & {param[2]}/{param[3]} &  {error:1.2e} &  {ls_nfe} + {newt_nfe} \n \t\t\t \\\\ \hline \n"

        file.write(row)

    file.write(" \end{tabular} \n \end{center} ")

    file.close()


if __name__ == "__main__":
    scipy_params = [
        ("RK45", 1e-6),
        ("Radau", 1e-6),
        ("BDF", 1e-6),
        ("LSODA", 1e-6),
        ("RK45", 1e-9),
        ("Radau", 1e-9),
        ("BDF", 1e-9),
        ("LSODA", 1e-9),
    ]
    pan_params = [
        (50, 0.3, 1e-6, 1e-6),
        (100, 0.3, 1e-6, 1e-6),
        (50, 0.3, 1e-9, 1e-9),
        (100, 0.3, 1e-9, 1e-9),
    ]

    t_lims = [0.0, 30.0]
    y_init = torch.rand(3) + 15

    # latex_table(
    #     lorenz,
    #     y_init,
    #     t_lims,
    #     scipy_params,
    #     pan_params=pan_params,
    #     filename="metrics2d.txt",
    # )

    lorenz_anim(t_lims, y_init)
