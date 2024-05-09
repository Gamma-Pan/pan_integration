import torch
from torch import tensor, nn
from pan_integration.solvers.pan_integration import pan_int
from pan_integration.utils.plotting import VfPlotter


class Spiral(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def forward(self, t, y):
        return (self.A @ y[..., None])[..., 0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b = 0.4, 1.0
    A = torch.tensor([[-a, b], [-b, -a]])
    f = Spiral(A).to(device)
    y_init = tensor([[1.4, 1.4]])

    plotter = VfPlotter(f=f)
    sol = plotter.solve_ivp(torch.linspace(0, 1, 100), y_init, set_lims=True)
    print(sol[-1])

    def callback(i, approx, Dapprox=None):
        plotter.approx(approx, 0., Dapprox=Dapprox,  marker='o', markersize=1.5, alpha=0.2)
        if i < 0:
            plotter.wait()
        else:
            plotter.fig.canvas.flush_events()
            plotter.fig.canvas.draw()

    plotter.wait()

    approx = pan_int(
        f,
        [0.0, 1.0],
        y_init,
        num_coeff_per_dim=10,
        num_points=100,
        optimizer_class=torch.optim.SGD,
        max_iters=10000,
        optimizer_params={"lr": 1e-6, "momentum": 0.9},
        callback=callback
    )

    print(approx[-1])
