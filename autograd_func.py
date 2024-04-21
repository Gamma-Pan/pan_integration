import torch
from torch import nn, tensor
from pan_integration.solvers.pan_integration import make_pan_adjoint
from torchdyn.core import NeuralODE


class NeurODE(nn.Module):
    def __init__(self, vf, num_coeff_per_dim, num_points):
        super().__init__()
        thetas = vf.parameters()
        self.pan_int = make_pan_adjoint(vf, thetas, num_coeff_per_dim, num_points)

    def forward(self, t, y_init):
        t_eval, traj = self.pan_int(y_init, t)
        return t_eval, traj


class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, t, y, *args, **kwargs):
        return self.seq(y)


if __name__ == "__main__":
    y_init = torch.rand(10, 2).requires_grad_(True)

    vf = VF()
    t_span = torch.linspace(0, 1, 5)

    model = NeurODE(vf, 15, 20)
    t_eval, traj = model(t_span, y_init )

    # model = NeuralODE(vf, sensitivity="adjoint")
    # t_eval, traj = model(y_init, t_span)

    yT = traj[-1]
    L = torch.nn.MSELoss()(yT, torch.ones_like(yT))
    L.backward()
