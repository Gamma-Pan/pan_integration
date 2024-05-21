import torch
from torch import tensor, nn
from pan_integration.numerics.functional import  pan_int
from pan_integration.numerics.pan_solvers import PanZero, T_grid
from pan_integration.utils.plotting import VfPlotter
import matplotlib.pyplot as plt
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint
from pan_integration.numerics.pan_solvers import PanZero

torch.random.manual_seed(42)



class Spiral(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return (self.A @ y[..., None])[..., 0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b = 0.4, 1.0
    A = torch.tensor([[-a, b], [-b, -a]])
    f = Spiral(A).to(device)
    y_init = 2 * torch.rand(1, 2) - 2
    t_lims = [0.0, 5.0]
    t_span = torch.linspace(*t_lims, 100)

    plotter = VfPlotter(f=f)
    sol = plotter.solve_ivp(
        t_span,
        y_init,
        set_lims=True,
        ivp_kwargs=dict(solver="tsit5", atol=1e-9),
    )

    def callback(B, t_lims):
        plotter.approx(
            B, t_lims,Dapprox=None, marker=None, markersize=1.5, alpha=0.9, color='green'
        )
        plotter.wait()
        # plotter.fig.canvas.flush_events()
        # plotter.fig.canvas.draw()

    plotter.wait()
    f.nfe = 0

    # Phi_plot = T_grid(torch.linspace(t_lims[0], t_lims[1], 100))

    solver = PanZero(f, 16,16, 1e-2, 30,device=device)
    approx, _, metrics = solver.solve(t_span, y_init)
    plotter.approx( approx, t_span[0])

    print(metrics)
    plt.show()

