import torch
from torch import tensor, nn
from pan_integration.core.pan_ode import PanSolver, T_grid
from pan_integration.utils.plotting import VfPlotter
import matplotlib.pyplot as plt
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint
from pan_integration.core.pan_ode import PanSolver
from pan_integration.core.solvers import PanSolver2


class Spiral(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return nn.functional.tanh(self.A @ y[..., None])[..., 0]


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    a, b = 0.4, 1.0
    A = torch.tensor([[-a, b], [-b, -a]]).to(device)
    f = Spiral(A).to(device)
    y_init = (2 * torch.rand(1, 2) - 2).to(device)
    t_lims = [0.0, 5.0]
    t_span = torch.linspace(*t_lims, 100)

    plotter = VfPlotter(f=f)
    sol_true = plotter.solve_ivp(
        t_span,
        y_init,
        set_lims=True,
        ivp_kwargs=dict(solver="tsit5", atol=1e-9, rtol=1e-9),
    )
    f.nfe = 0
    sol = plotter.solve_ivp(
        t_span,
        y_init,
        set_lims=False,
        ivp_kwargs=dict(solver="tsit5", atol=1e-4, rtol=1e-4),
        plot_kwargs=dict(color="orange", linestyle="--"),
    )
    plotter.wait()
    print(f"tsit | nfe: {f.nfe} | err: {torch.norm(sol_true-sol)} ")

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
        # plotter.ax.plot(approx[-1,0,0], approx[-1,0,1], 'o', color='lime',alpha=0.3, markersize=5)
        # plotter.wait()
        plotter.fig.canvas.flush_events()
        plotter.fig.canvas.draw()
        # plt.pause(0.1)

    f.nfe = 0

    solver = PanSolver2(32, 100, delta=1e-4 ,callback=callback)

    approx, B = solver.solve(f, t_span, y_init, B_init=None)
    plotter.approx(B, t_lims, y_init, show_arrows=False )

    print(f"pan | nfe: {f.nfe} | err: {torch.norm(approx-sol_true)} ")


    plt.show()
