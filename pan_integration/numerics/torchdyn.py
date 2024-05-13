from .functional import pan_int
from torchdyn.numerics.solvers.ode import SolverTemplate
import torch

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
