import torch
from torch import tensor, nn, linalg
from torch.nn import functional as F
from pan_integration.core.pan_ode import PanSolver, T_grid
from pan_integration.utils.plotting import VfPlotter, wait
import matplotlib.pyplot as plt
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint

from matplotlib.animation import FFMpegFileWriter

torch.manual_seed(23)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class NN(nn.Module):
    def __init__(self, std=2.0):
        super().__init__()
        self.w1 = torch.nn.Linear(2, 5)
        torch.nn.init.normal_(self.w1.weight, std=std)
        self.w2 = torch.nn.Linear(5, 5)
        torch.nn.init.normal_(self.w2.weight, std=std)
        self.w3 = torch.nn.Linear(5, 2)
        torch.nn.init.normal_(self.w3.weight, std=std)
        self.A = torch.tensor([[-0.9, -2.0], [1.5, -1]], device=device)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        y = torch.cos(0.5 * self.w1(y))
        y = F.softplus(0.5 * self.w2(y))
        y = F.tanh(self.w3(y))
        return F.tanh(self.A @ y[..., None]).squeeze(-1)


if __name__ == "__main__":

    f = NN(std=5.0).to(device)
    for param in f.parameters():
        param.requires_grad_(False)

    y_init = 1 * torch.randn(5, 2, device=device)

    t_lims = [0, 2]

    plotter = VfPlotter(f, grid_definition=64, animation=True)
    sol_true = plotter.solve_ivp(
        torch.linspace(*t_lims, 100),
        y_init,
        set_lims=True,
        ivp_kwargs=dict(solver="tsit5", atol=1e-9, rtol=1e-9),
        plot_kwargs=dict(alpha=0.2, color="green"),
    )
    sol_true = sol_true.to(device)
    f.nfe = 0
    _, sol = odeint(
        f,
        y_init,
        t_span=torch.linspace(*t_lims, 2),
        solver="tsit5",
        atol=1e-4,
        rtol=1e-4,
        return_all_eval=True,
    )
    plotter.ax.plot(
        sol[:, :, 0].cpu(),
        sol[:, :, 1].cpu(),
        "--",
        color="cyan",
        alpha=0.2,
        label="Tsitouras5",
    )
    plotter.wait()
    print(f"tsit | nfe: {f.nfe} | err: {torch.norm(sol_true[-1]-sol[-1])} ")

    t1 = plt.text(
        s=f"nfe : {f.nfe} | err : {torch.norm(sol_true[-1]-sol[-1]):.5f}",
        x=0,
        y=0,
        color="blue",
        transform=plotter.ax.transAxes,
    )
    t2 = plt.text(s="test", x=0, y=1, color="green", transform=plotter.ax.transAxes)

    plotter.wait()

    # writer = FFMpegFileWriter(fps=4, metadata=dict(title="ode sol"))
    # writer.setup(plotter.fig, "output_arr.mp4", dpi=100)
    # writer.frame_format = "png"

    def callback(i, tt_lims, y_init, f_init, B, lr):
        approx = plotter.approx(
            tensor(t_lims),
            B,
            num_arrows=0,
            num_points=101,
            marker=None,
            markersize=2.5,
            alpha=0.70,
            color="green",
        )

        err = torch.norm(approx[-1] - sol_true[-1])

        t2.set_text(f"nfe: {f.nfe} | err: {err:.5f}") # | lr {lr:.5f}")

        plotter.fig.canvas.flush_events()
        plotter.fig.canvas.draw()
        # plt.pause(0.2)

        if i > 100:
            plotter.wait()

        # writer.grab_frame()

    f.nfe = 0

    solver = PanSolver(
        num_coeff_per_dim=32,
        callback=callback,
        device=device,
        tol=1e-3,
        min_lr=1e-3,
        max_iters=1000,
        gamma=0.9,
    )

    approx, _ = solver.solve(f, torch.linspace(*t_lims, 2, device=device), y_init)

    print(f" pan | nfe: {f.nfe} | err: {torch.norm(approx[-1] - sol_true[-1])} ")

    # writer.finish()
    plotter.wait()
