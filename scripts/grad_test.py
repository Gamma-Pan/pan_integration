import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pan_integration.utils.plotting
from pan_integration.core.pan_ode import PanODE, PanSolver, PanSolver
from torchdyn.models import NeuralODE
import torchdyn
from pan_integration.utils.plotting import wait

torch.manual_seed(28)

DIM = 2

class NN(nn.Module):
    def __init__(self, wstd=2.0):
        super().__init__()
        self.w1 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w1.weight, std=wstd)
        self.w2 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w2.weight, std=wstd)
        self.w3 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w3.weight, std=wstd)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = F.tanh(self.w1(x))
        x = F.tanh(self.w2(x))
        x = F.tanh(self.w3(x))
        x = x
        return x


coeff = 32
coeff_adj = 64
steps = 4


if __name__ == "__main__":
    vf = NN(wstd=0.5)

    y_init = torch.rand(4, DIM)
    t_span = torch.linspace(0, 1, steps)

    vf.zero_grad()
    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-4, atol_adjoint=1e-4
    )

    traj = ode_model(y_init, t_span)

    L = torch.sum((traj[-1] - 2 * torch.ones_like(y_init)) ** 2)
    L.backward()

    grads = [w.grad for w in vf.parameters()].copy()

    vf.zero_grad()

    print("-=-=-=|*|=-=-=-")

    solver_conf = dict(num_coeff_per_dim=coeff, max_iters=50, delta=1e-4)
    solver_conf_adjoint = dict(num_coeff_per_dim=coeff_adj, max_iters=50, delta=1e-4)

    solver = PanSolver(**solver_conf, callback=None)
    solver_adjoint = PanSolver(**solver_conf_adjoint, callback=None)

    pan_ode_model = PanODE(vf, solver, solver_adjoint, sensitivity="adjoint")

    _, traj_pan = pan_ode_model(y_init, t_span)

    L_pan = torch.sum((traj_pan[-1] - 2 * torch.ones_like(y_init)) ** 2)

    L_pan.backward()

    grads_pan = [w.grad for w in vf.parameters()].copy()

    print("-+=+-")

    print("SOLUTON \n")
    print(traj[-1])
    print(traj_pan[-1], "\n")

    print("GRADS\n")
    print(torch.norm(grads[0] - grads_pan[0]), "\n")
    print(torch.norm(grads[1] - grads_pan[1]), "\n")
    print(torch.norm(grads[2] - grads_pan[2]), "\n")
