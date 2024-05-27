import torch
from torch import nn
from pan_integration.core.ode import PanZero, make_pan_adjoint
from torchdyn.models import NeuralODE



class NN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand(10, 10))
        self.w2 = torch.nn.Parameter(torch.rand(10, 10))
        self.w3 = torch.nn.Parameter(torch.rand(10, 10))
        self.nfe = 0

    def forward(self, t, y, *args, **kwargs):
        self.nfe += 1
        return torch.tanh(self.w3 @ torch.tanh(self.w2 @ torch.tanh(self.w1 @ y)))


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        solver,
        solver_adjoint,
    ):
        super().__init__()
        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.solver = solver
        self.solver_adjoint = solver_adjoint

        self.pan_int = make_pan_adjoint(
            self.vf,
            self.thetas,
            self.solver,
            self.solver_adjoint,
        )

    def forward(self, y_init, t_eval, *args, **kwargs):
        traj, B = self.solver.solve(self.vf, t_eval, y_init)
        return t_eval, traj,


if __name__ == "__main__":
    vf = NN()

    y_init = torch.rand(10, 10, 10)
    t_span = torch.linspace(0, 1, 2)

    solver = PanZero(8, 8)
    solver_adjoint = PanZero(10, 10)

    pan_ode_model = PanODE(vf, solver, solver_adjoint)
    _, traj_pan, _ = pan_ode_model(y_init, t_span)
    L_pan = torch.sum((traj_pan[-1] - 1 * torch.ones(10, 10)) ** 2)
    L_pan.backward()
    grads_pan = [w.grad for w in vf.parameters()]

    vf.zero_grad()

    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-9, atol_adjoint=1e-9
    )
    traj = ode_model(y_init, t_span)
    L = torch.sum((traj[-1] - 1 * torch.ones(10, 10)) ** 2)
    L.backward()
    grads = [w.grad for w in vf.parameters()]

    print("SOLUTION \n")
    print(traj[-1], "\n", traj_pan[-1], "\n")

    print("GRADS\n")
    print(torch.norm(grads[0]- grads_pan[0]), "\n")
    print(torch.norm(grads[1]- grads_pan[1]), "\n")
    print(torch.norm(grads[2]- grads_pan[2]), "\n")

    # print(grads_pan[2]/ grads[2])
