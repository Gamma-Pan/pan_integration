import torch
from torch import tensor, nn
from pan_integration.core.functional import pan_int
from pan_integration.core.ode import PanSolver, T_grid
from pan_integration.utils.plotting import VfPlotter
import matplotlib.pyplot as plt
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint
from pan_integration.core.ode import PanSolver


class Spiral(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return torch.tanh(self.A @ y[..., None])[..., 0]


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
    sol = plotter.solve_ivp(
        t_span,
        y_init,
        set_lims=True,
        ivp_kwargs=dict(solver="tsit5", atol=1e-9),
    )

    def callback(t_lims, y_init, B):
        plotter.approx(
            B,
            t_lims,
            show_arrows=True,
            marker=None,
            markersize=1.5,
            alpha=0.9,
            color="green",
        )
        plotter.wait()
        # plotter.fig.canvas.flush_events()
        # plotter.fig.canvas.draw()

    plotter.wait()
    f.nfe = 0

    optim = {"optimizer_class": torch.optim.RMSprop, "params": {"lr": 1e-2}}
    solver = PanSolver(
        32,
        32,
        1e-2,
        30,
        device=device,
        callback=callback,
        optim=optim,
    )
    approx = solver.solve(f, t_span, y_init)
    print(f.nfe)
    plt.show()
