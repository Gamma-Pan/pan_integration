import torch
from torch import nn
from pan_integration.core.pan_ode import PanODE, PanSolver
from torchdyn.models import NeuralODE
from pan_integration.utils.plotting import wait

torch.manual_seed(42)

DIM = 10


class NN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(DIM, DIM))
        self.w2 = torch.nn.Parameter(torch.randn(DIM, DIM))
        self.w3 = torch.nn.Parameter(torch.randn(DIM, DIM))
        self.nfe = 0

    def forward(self, t, y, *args, **kwargs):
        self.nfe += 1
        return torch.tanh(self.w3 @ torch.tanh(self.w2 @ torch.tanh(self.w1 @ y)))


import matplotlib.pyplot as plt
from pan_integration.core.solvers import T_grid


if __name__ == "__main__":
    vf = NN()

    y_init = torch.rand(1, DIM, DIM)
    t_span = torch.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    (line,) = ax.plot(t_span, torch.zeros_like(t_span))

    xs = torch.linspace(0,1,100 )
    ys = torch.linspace(-1, 1, 100)
    Xs,Ys = torch.meshgrid(xs,ys)
    batch = torch.stack((Xs, Ys), dim=-1).reshape(-1, 2)
    derivatives = vf(0, batch).reshape(100, 100, 10,10)
    Us, Vs = derivatives.unbind(dim=-1)
    ax.streamplot( Xs, Ys, Us, Vs, )


    def callback(t, y, B):
        dims = len(B.shape) - 1
        num_coeff = B.shape[-1]
        t_in = torch.linspace(*t, 100)
        t_out = -1 + 2 * (t_in - t_in[0]) / (t_in[-1] - t_in[0])
        Phi = T_grid(t_out, num_coeff)
        approx = (B @ Phi)[*(0 for _ in range(dims)), :]
        line.set_data(t_in, approx)
        wait()

    optim = {"optimizer_class": torch.optim.SGD, "params": {"lr": 1e-9}}
    deltas = (1e-3, 1e-5)
    max_iters = (20, 0)
    solver_conf = dict(
        num_points=100,
        num_coeff_per_dim=32,
        optim=optim,
        deltas=deltas,
        max_iters=max_iters,
    )
    solver = PanSolver(**solver_conf)
    solver_adjoint = PanSolver(**solver_conf, callback=callback)

    pan_ode_model = PanODE(vf, t_span, solver, solver_adjoint)
    _, traj_pan = pan_ode_model(y_init, t_span)

    L_pan = torch.sum((traj_pan[-1] - 1 * torch.rand(DIM, DIM)) ** 2)
    L_pan.backward()
    grads_pan = [w.grad for w in vf.parameters()]

    vf.zero_grad()

    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-9, atol_adjoint=1e-9
    )
    traj = ode_model(y_init, t_span)
    L = torch.sum((traj[-1] - 1 * torch.ones(DIM, DIM)) ** 2)
    L.backward()
    grads = [w.grad for w in vf.parameters()]

    print("SOLUTION \n")
    print(torch.norm(traj[-1] - traj_pan[-1]), "\n")

    print("GRADS\n")
    print(torch.norm(grads[0] - grads_pan[0]), "\n")
    print(torch.norm(grads[1] - grads_pan[1]), "\n")
    print(torch.norm(grads[2] - grads_pan[2]), "\n")

    # print(grads_pan[2]/ grads[2])
