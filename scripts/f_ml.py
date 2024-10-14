import torch
from torch import tensor, nn, linalg, cos, sin, cosh, exp
import lightning
from torch.func import vmap

import plotly.express as px
import plotly.figure_factory as ff

import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, Patch

lightning.seed_everything(42)
from torchdyn.numerics import odeint

# %%

DIMS = 2
NUM_COFF = 20
device = torch.device("cpu")


class NN(nn.Module):
    def __init__(self, std=5.0, hidden_dim=10):
        super().__init__()
        self.w1 = torch.nn.Linear(DIMS, hidden_dim)
        torch.nn.init.normal_(self.w1.weight, std=std)
        self.w2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.normal_(self.w2.weight, std=std)
        self.w3 = torch.nn.Linear(hidden_dim, DIMS)
        torch.nn.init.normal_(self.w3.weight, std=std)

        self.nfe = 0

    def forward(self, t, y):
        y = torch.tanh(self.w1(y))
        y = torch.tanh(self.w2(y))
        y = torch.cos(torch.pi * self.w3(y))
        return y

class Fun(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.nfe = 0
        self.c = c

    def forward(self, t, y):
        self.nfe += 1
        return -self.c * y + self.c * sin(y)


class LTV(nn.Module):
    def __init__(self, num_params=5, dims=DIMS):
        super().__init__()
        self.A = torch.rand(DIMS, DIMS*params)

# %% plotter class


class ModelPlotter:
    def __init__(self, f, model, scale=0.01, definition=10):
        self.fig = go.Figure()

        self.f = f
        self.model = model

        self.scale = scale
        self.definition = definition

    def _quiver(self, x, y, u, v, id, q_args=None):

        quiver_args = dict(
            marker=dict(symbol="arrow-up", angleref="previous", color="blue"),
            line=dict(color="blue"),
            opacity=0.5,
        )

        if q_args is not None:
            for k, v in q_args.items():
                quiver_args[k] = v

        for idx, (x_i, y_i, u_i, v_i) in enumerate(zip(x, y, u, v)):
            self.fig.add_trace(
                go.Scatter(
                    x=[x_i, x_i + self.scale * u_i],
                    y=[y_i, y_i + self.scale * v_i],
                    mode="lines+markers",
                    customdata=(id, idx),
                    name="arrows",
                    legendgroup=id,
                    showlegend=idx == 0,
                    **quiver_args
                )
            )

    def _make_vf(self, func, t, tid=None, xlims=(-1, 1), ylims=(-1, 1), q_args=None):
        if q_args is None:
            q_args = dict()
        if tid is None:
            tid = str(torch.rand(1).item())[2:]

        x = torch.linspace(*xlims, self.definition)
        y = torch.linspace(*ylims, self.definition)

        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.stack([X, Y]).reshape(2,-1).T
        DZ = func(..., Z)

        self._quiver(
            *Z.unbind(-1),
            *DZ.unbind(-1),
            tid,
            q_args
        )


# %%
f = NN(std=0.5)
for parameter in f.parameters():
    parameter.requires_grad_(False)

plotter = ModelPlotter(
    f,
    ...,
    scale=0.2,
    definition=40,
)
plotter._make_vf(plotter.f, ..., xlims=(-5,5), ylims=(-5,5))
plotter.fig.show()
