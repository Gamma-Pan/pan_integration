import plotly.subplots
import torch
from torch import tensor
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from torchmetrics.utilities.plot import plot_curve

from pan_integration.core.solvers import T_grid, DT_grid
from torchdyn.numerics import odeint
import plotly.figure_factory as ff

# %% constants

t_lims = [-1.0, 1.0]
t = torch.linspace(*t_lims, 100)
num_coeff = 10
y_init = tensor([-2.0])


# %% f, solution

from torch import nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, std=5.0, dims=1, hidden_dims=100):
        super().__init__()

        self.w1 = torch.nn.Linear(dims, hidden_dims)
        torch.nn.init.normal_(self.w1.weight, std=std)

        self.w2 = torch.nn.Linear(hidden_dims, hidden_dims)
        torch.nn.init.normal_(self.w2.weight, std=std)

        self.w3 = torch.nn.Linear(hidden_dims, dims)
        torch.nn.init.normal_(self.w3.weight, std=std)

        self.A = torch.tensor([[-0.9, -2.0], [1.5, -1]])
        self.nfe = 0

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, t, y):
        self.nfe += 1
        # y = self.w1(y + t)
        # y = torch.tanh(self.w2(y))
        # y = torch.cos(2*torch.pi*t*self.w3(y))
        return 5 * torch.cos(2 * torch.pi * t + y) + t


f = NN(std=5)


# %% evaluator


class Integrator:
    def __init__(self, y_init, f_init, t_lims, num_points=100, num_coeff=10):

        self.y_init = y_init
        self.f_init = f_init
        self.dt = 2 / (t_lims[1] - t_lims[0])
        self.a = 1 / self.dt
        t_scld = torch.linspace(-1, 1, num_points)

        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (t_scld + 1)

        self.Phi = T_grid(t_scld, num_coeff)
        self.DPhi = DT_grid(t_scld, num_coeff)

        DPhi_r = self.DPhi[2:, 1:] - self.DPhi[2:, [0]]
        DPhi_r_inv = torch.linalg.pinv(DPhi_r)

        self.Phi_r = (
            -self.Phi[2:, [0]]
            - self.DPhi[2:, [0]]
            - self.DPhi[2:, [0]] * t_scld[None, 1:]
            + self.Phi[2:, 1:]
        )
        Phi_r_inv = torch.linalg.pinv(self.Phi_r)

        self.C = self.a * f_init + y_init + self.a * f_init * t_scld[None, 1:]

    def add_head(self, b):
        Phi_R0 = self.Phi[2:, [0]]
        DPhi_R0 = self.DPhi[2:, [0]]

        phi_inv = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        b_01 = (
            torch.stack([self.y_init, (1 / self.dt) * self.f_init], dim=-1)
            - b @ torch.hstack([Phi_R0, DPhi_R0])
        ) @ phi_inv

        return torch.cat([b_01[0], b], dim=-1)

    def b2func(self, bm2):
        b = self.add_head(bm2)
        return b @ self.Phi, b @ self.DPhi

    def step(self, b, t):



        y_k = self.C + b @ self.Phi_r
        f_k = f( t[:,None], y_k[:,None] )[:,0]

        return self.dt



# %% solution
t_sol, solution = odeint(
    f, y_init, tensor(t_lims), "tsit5", return_all_eval=True, atol=1e-6, rtol=1e-6
)
sol_min = (smi := solution.min().item()) - 2 * abs(smi)
sol_max = (sma := solution.max().item()) + 2 * abs(sma)


# %% streamplot

grid_res = 10
t_grid = torch.linspace(*t_lims, grid_res)
y_grid = torch.linspace(sol_min, sol_max, grid_res)

T, Y = torch.meshgrid(t_grid, y_grid, indexing="xy")
derivatives = f(T.reshape(-1)[:, None], Y.reshape(-1)[:, None])
U = torch.ones_like(T)
V = derivatives.reshape_as(Y)

streamplot = ff.create_streamline(
    t_grid, y_grid, U, V, arrow_scale=0.1, line=dict(color="gray"), angle=torch.pi / 30
)
streamplot.update_layout(template="plotly_white")
streamplot.update_traces(opacity=0.4)
streamplot.add_trace(
    go.Scatter(x=t, y=solution.view(-1), mode="lines", line=dict(color="teal")),
)

# %% quiver

torch.manual_seed(42)
f_init = f(t_lims[0], y_init)
int = Integrator(y_init, f_init, t_lims, num_coeff=10)

fig = make_subplots(rows=1, cols=2)
fig.update_layout(template="plotly_white")

fig.add_trace(go.Scatter(x=t_sol, y=solution[:, 0], mode="lines"), row=1, col=2)

b_dot = torch.zeros(num_coeff)
DPhi_small = DT_grid(t[::10] ,10)

num_steps = 10
for i in range(num_steps):
    b = torch.randn(num_coeff - 2) * torch.pow(10, -torch.arange(num_coeff - 2) / 10)
    y_b, Dy_b = int.b2func(b)
    fig.add_trace(
        go.Scatter(
            x=t, y=y_b, mode="lines", line=dict(color="red", width=1.4), legendgroup=i
        ),
        row=1,
        col=1,
    )

    num_arrows = y_b.shape

    q_t = t[::10]
    q_y = y_b[::10]
    q_Dy = Dy_b[::10]
    derivatives = f(t[:, None], y_b.view(-1)[:, None])
    q_derivatives = derivatives[::10, 0]
    uD = uf = torch.ones_like(q_y)

    quiver_Dy = ff.create_quiver(
        x=q_t,
        y=q_y,
        u=uD,
        v=q_Dy,
        scale=0.1,
        arrow_scale=0.1,
        line=dict(color="red"),
        legendgroup=i,
    )
    quiver_fy = ff.create_quiver(
        x=q_t,
        y=q_y,
        u=uf,
        v=q_derivatives,
        scale=0.1,
        arrow_scale=0.1,
        line=dict(color="gray"),
        legendgroup=i,
    )

    # fig.add_traces(quiver_Dy.data)
    fig.add_traces(quiver_fy.data, rows=1, cols=1)

    b_dot = ((q_Dy - q_derivatives)@torch.linalg.pinv(DPhi_small) - b_dot) / (i + 1)

    fig.add_trace(
        go.Scatter(
            x=q_t,
            y=e,
            mode="lines",
            line=dict(color="cyan"),
            opacity=0.1,
        ),
        row=1,
        col=2,
    )

fig.show()
