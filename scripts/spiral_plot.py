import torch
from torch import tensor, nn
from torch.nn import functional as F
from pan_integration.core.pan_ode import PanSolver, T_grid
from pan_integration.utils.plotting import VfPlotter
import matplotlib.pyplot as plt
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint

torch.manual_seed(23)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN(nn.Module):
    def __init__(self, std=2.0):
        super().__init__()
        # self.w1 = torch.nn.Linear(2, 2)
        # torch.nn.init.normal_(self.w1.weight, std=std)
        # self.w2 = torch.nn.Linear(2, 2)
        # torch.nn.init.normal_(self.w2.weight, std=std)
        # self.w3 = torch.nn.Linear(2, 2)
        # torch.nn.init.normal_(self.w3.weight, std=std)
        self.A = torch.tensor([[-0.9, -2.], [1.5, -1]])
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        # y = torch.cos(0.5*self.w1(y))
        # y = F.softplus(0.5*self.w2(y))
        # y = F.tanh(self.w3(y))
        return F.tanh(self.A @ y[...,None]).squeeze(-1)


if __name__ == "__main__":

    f = NN(std=3.4).to(device)
    for param in f.parameters():
        param.requires_grad_(False)

    y_init = (2 * torch.randn(2, 2, device=device))
    t_lims = [0, 5]

    plotter = VfPlotter(f=f, grid_definition=(100, 100))
    sol_true = plotter.solve_ivp(
        torch.linspace(*t_lims, 100),
        y_init,
        set_lims=True,
        ivp_kwargs=dict(solver="tsit5", atol=1e-9, rtol=1e-9),
        plot_kwargs=dict(alpha=0.5),
    )
    f.nfe = 0
    _, sol = odeint(
        f,
        y_init,
        t_span=torch.linspace(*t_lims, 2),
        solver="tsit5",
        atol=1e-3,
        rtol=1e-3,
        return_all_eval=True,
    )
    plotter.ax.plot(sol[:, :, 0], sol[:, :, 1], "--", color="cyan")
    plotter.wait()
    print(f"tsit | nfe: {f.nfe} | err: {torch.norm(sol_true[-1]-sol[-1])} ")
    max_iter = f.nfe

    def callback(t_lims, y_init, B):
        approx = plotter.approx(
            B,
            t_lims,
            y_init,
            show_arrows=False,
            marker=None,
            markersize=1.5,
            alpha=0.9,
            color="green",
        )

        plotter.ax.plot(
            approx[0, :, 0],
            approx[0, :, 1],
            "o",
            color="darkgreen",
            alpha=0.3,
            markersize=5,
        )
        # plotter.wait()
        # plotter.fig.canvas.flush_events()
        # plotter.fig.canvas.draw()
        plt.pause(0.3)

    f.nfe = 0

    solver = PanSolver(
        16,
        # num_points=64 - 2,
        max_iters=int(max_iter * 1),
        delta=1e-4,
        callback=callback,
    )

    approx = solver.solve(f, torch.linspace(*t_lims, 5), y_init, B_init=None)

    print(f"pan | nfe: {f.nfe} | err: {torch.norm(approx[-1]-sol_true[-1])} ")

    print(f"true {sol_true[-1]} \n", f"tsit {sol[-1]} \n", f"pan {approx} \n")

    plotter.wait()
