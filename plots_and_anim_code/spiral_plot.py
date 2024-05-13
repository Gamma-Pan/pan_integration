import torch
from torch import tensor, nn
from pan_integration.numerics.functional import  pan_int
from pan_integration.utils.plotting import VfPlotter
import matplotlib.pyplot as plt
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.core.neuralde import odeint

torch.random.manual_seed(42)

class PanSolver(SolverTemplate):
    def __init__(
            self,
            dtype=torch.float32,
            num_coeff_per_dim=16,
            num_points=64,
            tol_zero = 1e-3,
            tol_one = 1e-5,
            max_iters_zero=30,
            max_iters_one=10,
            optimizer_class=None,
            optimizer_params=None,
            init="random",
            coarse_steps=5,
            callback=None
    ):
        super().__init__(order=0)
        self.dtype = dtype
        self.stepping_class = "fixed"

        self.num_coeff_per_dim = num_coeff_per_dim
        self.num_points = num_points
        self.tol_zero = tol_zero
        self.tol_one = tol_one
        self.max_iters_zero = max_iters_zero
        self.max_iters_one = max_iters_one
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.init = init
        self.coarse_steps = coarse_steps
        self.callback = callback

    def step(self, f, y_init, t, dt, k1=None, args=None):
        approx = pan_int(
            f,
            torch.tensor([t, t+dt]),
            y_init,
            self.num_coeff_per_dim,
            self.num_points,
            self.tol_zero,
            self.tol_one,
            self.max_iters_zero,
            self.max_iters_one,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.optimizer_params,
            init=self.init,
            coarse_steps=self.coarse_steps,
            callback = self.callback
        )
        return None, approx[-1], None

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
    y_init = 2 * torch.rand(10, 2) - 2
    t_lims = [0.0, 5.0]
    t_span = torch.linspace(*t_lims, 100)

    plotter = VfPlotter(f=f)
    sol = plotter.solve_ivp(
        t_span,
        y_init,
        set_lims=True,
        ivp_kwargs=dict(solver="tsit5", atol=1e-3),
    )
    print(f.nfe)

    def callback( t, approx, Dapprox=None):
        plotter.approx(
            approx, t, Dapprox=None, marker=None, markersize=1.5, alpha=0.9, color='green'
        )
        # plotter.wait()
        plotter.fig.canvas.flush_events()
        plotter.fig.canvas.draw()

    plotter.wait()
    print(f.nfe)
    f.nfe = 0

    solver = PanSolver(
        dtype = torch.float32,
        num_coeff_per_dim=32,
        num_points=32,
        max_iters_zero=30,
        max_iters_one=30,
        init='euler',
        coarse_steps=5,
        optimizer_class=torch.optim.SGD,
        optimizer_params={"lr": 1e-9 ,"momentum": 0.95, "nesterov": True},
        tol_zero = 1e-3,
        tol_one=1,
        callback=callback,
    )

    _, torchsol,  = odeint(f, y_init, torch.linspace(*t_lims, steps=5), solver= solver)
    print(f.nfe)

    plt.show()

    print(f.nfe)
