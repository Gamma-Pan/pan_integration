import torch
from torch import nn
from pan_integration.core.pan_ode import PanODE, PanSolver
from torchdyn.models import NeuralODE
from pan_integration.utils.plotting import wait

torch.manual_seed(42)

DIM = 4


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
        self.tanh = torch.nn.Tanh()
        self.nfe = 0

    def forward(self, t, y, *args, **kwargs):
        self.nfe += 1
        x = self.tanh(self.w1(y))
        x = self.tanh(self.w2(y))
        x = self.tanh(self.w3(y))
        return x


import matplotlib.pyplot as plt
from pan_integration.core.solvers import T_grid, DT_grid
from math import ceil, floor, sqrt


if __name__ == "__main__":
    vf = NN()

    y_init = torch.rand(1, DIM)
    POINTS = 128
    t_span = torch.linspace(0, 1, POINTS)

    fig, axes = plt.subplots(ceil(sqrt(DIM)), ceil(sqrt(DIM)))
    lines = []
    for ax in axes.reshape(-1):
        lines.append(ax.plot(t_span, torch.zeros_like(t_span), color='red')[0])

    def callback(t, y, B):
        dims = len(B.shape) - 1
        num_coeff = B.shape[-1]
        t_in = torch.linspace(*t, POINTS)
        t_out = -1 + 2 * (t_in - t_in[0]) / (t_in[-1] - t_in[0])
        Phi = T_grid(t_out, num_coeff)
        approx = (B @ Phi)[0, :, :]

        for line, data in zip(lines, approx):
            line.set_data(t_in, data)

        for ax in axes.reshape(-1):
            ax.relim()
            ax.autoscale_view()

        fig.canvas.flush_events()
        fig.canvas.draw()
        # wait()

    optim = {"optimizer_class": torch.optim.Adam, "params": {"lr": 1e-4}}
    max_iters = (20, 0)
    solver_conf = dict(
        num_points=POINTS,
        num_coeff_per_dim=32,
        optim=optim,
        deltas=(1e-2,1e-3 ),
        max_iters=max_iters,
    )
    solver_conf_adjoint = dict(
        num_points=POINTS,
        num_coeff_per_dim=32,
        optim=optim,
        deltas=(1e-2, 1e-3),
        max_iters=(1, int(30)),
    )
    solver = PanSolver(**solver_conf)
    solver_adjoint = PanSolver(**solver_conf_adjoint, callback=callback)

    pan_ode_model = PanODE(vf, t_span, solver, solver_adjoint, sensitivity="adjoint")
    _, traj_pan = pan_ode_model(y_init, t_span)

    L_pan = torch.sum((traj_pan[-1] - 1 * torch.ones_like(y_init)) ** 2)
    L_pan.backward()
    grads_pan = [w.grad for w in vf.parameters()]

    vf.zero_grad()

    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-9, atol_adjoint=1e-9
    )
    traj = ode_model(y_init, t_span)
    L = torch.sum((traj[-1] - 1 * torch.ones_like(y_init)) ** 2)
    L.backward()
    grads = [w.grad for w in vf.parameters()]

    fig, ax = plt.subplots()
    ax.plot(t_span, traj_pan[:, 0, 3].detach(), "r")
    ax.plot(t_span, traj[:, 0, 3].detach(), "g")
    plt.show()

    print("SOLUTION \n")
    print(torch.norm(traj[-1] - traj_pan[-1]), "\n")

    print("GRADS\n")
    print(grads[0], "\n", grads_pan[0], "\n")
    print(grads[1], "\n", grads_pan[1], "\n")
    print(grads[2], "\n", grads_pan[2], "\n")

    # print(grads_pan[2]/ grads[2])
