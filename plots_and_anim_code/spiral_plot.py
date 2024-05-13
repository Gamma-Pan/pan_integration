import torch
from torch import tensor, nn
from pan_integration.solvers.pan_integration import  pan_int
from pan_integration.utils.plotting import VfPlotter
import matplotlib.pyplot as plt

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

    def callback( approx, Dapprox=None):
        plotter.approx(
            approx, 0.0, Dapprox=None, marker=None, markersize=1.5, alpha=0.9, color='green'
        )
        plotter.wait()
        # plotter.fig.canvas.flush_events()
        # plotter.fig.canvas.draw()

    plotter.wait()
    print(f.nfe)
    f.nfe = 0

    approx = pan_int(
        f,
        t_span,
        y_init,
        num_coeff_per_dim=32,
        num_points=32,
        max_iters_zero=30,
        max_iters_one=30,
        init='euler',
        coarse_steps=5,
        optimizer_class=torch.optim.SGD,
        optimizer_params={"lr": 1e-9 ,"momentum": 0.95, "nesterov": True},
        tol_zero = 1e-3,
        tol_one=1e-3,
        callback=callback,
    )
    print(approx.shape)
    plotter.approx( approx, t_init=0.0,  color='lime')

    print(sol[-1])
    print(approx[-1])

    plt.show()

    print(f.nfe)
