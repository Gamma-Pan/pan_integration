import torch
from torch import nn
import torch.nn.functional as F
from pan_integration.core.pan_ode import PanODE, PanSolver, PanSolver
from pan_integration.core.solvers import PanSolver2
from torchdyn.models import NeuralODE
from pan_integration.utils.plotting import wait

# torch.manual_seed(634)

DIM = 9
factor = 3

class NN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.w1 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w1.weight, std=1)
        self.w2 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w2.weight, std=1)
        self.w3 = torch.nn.Linear(DIM, DIM)
        torch.nn.init.normal_(self.w3.weight, std=1)
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = F.tanh(factor * self.w1(x))
        x = F.tanh(factor * self.w2(x))
        x = F.tanh(factor * self.w3(x))
        return x


import matplotlib.pyplot as plt
from pan_integration.core.solvers import T_grid, DT_grid
from math import ceil, floor, sqrt


if __name__ == "__main__":
    vf = NN()

    y_init = torch.rand(1, DIM)
    POINTS = 100
    t_span = torch.linspace(0, 1, POINTS)

    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-4, atol_adjoint=1e-4
    )
    traj = ode_model(y_init, t_span)
    L = torch.sum((traj[-1] - 2 * torch.ones_like(y_init)) ** 2)
    L.backward()
    grads = [w.grad for w in vf.parameters()]

    fig1, axes = plt.subplots(3, 3)
    for ax, data in zip(fig1.axes, traj[:, 0, :].mT):
        ax.plot(t_span, data.detach(), "g")

    lines = []
    for ax in axes.reshape(-1):
        lines.append(ax.plot(t_span, torch.zeros_like(t_span), "r")[0])
    wait()

    def callback(t, y, B):
        num_coeff = B.shape[-1]
        t_in = torch.linspace(*t, POINTS)
        t_out = -1 + 2 * (t_in - t_in[0]) / (t_in[-1] - t_in[0])
        Phi = T_grid(t_out, num_coeff)
        approx = y[0,...,None] + (B @ Phi)[0, :, :]

        for line, data in zip(lines, approx):
            line.set_data(t_in, data)

        # for ax in axes.reshape(-1):
        #     ax.relim()
        #     ax.autoscale_view()
        #
        # fig1.canvas.flush_events()
        # fig1.canvas.draw()
        plt.pause(0.01)
        # wait()


    solver_conf = dict(
        num_coeff_per_dim=32,
        max_iters=50,
        delta = 1e-4,
    )
    solver_conf_adjoint = dict(
        num_coeff_per_dim=32,
        max_iters=50,
        delta =1e-4
    )
    solver = PanSolver2(**solver_conf, callback=callback)
    solver_adjoint = PanSolver2(**solver_conf_adjoint, callback=callback)
    pan_ode_model = PanODE(vf, t_span, solver, solver_adjoint, sensitivity="adjoint")
    _, traj_pan = pan_ode_model(y_init, t_span)

    print('-+=+-')
    L_pan = torch.sum((traj_pan[-1] - 2 * torch.ones_like(y_init)) ** 2)
    L_pan.backward()
    grads_pan = [w.grad for w in vf.parameters()]

    vf.zero_grad()

    print("SOLUTION \n")
    print(torch.norm(traj[-1] - traj_pan[-1]), "\n")

    # print("GRADS\n")
    print(torch.norm(grads[0] - grads_pan[0]).item(), "\n")
    print(torch.norm(grads[1] - grads_pan[1]).item(), "\n")
    print(torch.norm(grads[2] - grads_pan[2]).item(), "\n")
    print(torch.norm(grads[3] - grads_pan[3]).item(), "\n")
    print(torch.norm(grads[4] - grads_pan[4]).item(), "\n")
    print(torch.norm(grads[5] - grads_pan[5]).item(), "\n")
