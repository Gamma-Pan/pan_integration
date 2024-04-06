import torch
from torch import nn
import torchdyn
from pan_integration.solvers import pan_int


if __name__ == '__main__':
    device = torch.device("cuda")

    net = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 10), nn.ReLU(), nn.BatchNorm1d(10)
    ).to(device)

    x = torch.rand(1,10)

    def f(batch):
        return batch

    y_init = torch.rand(4).to(device)
    t_lims = [0,1]
    y = pan_int(f, y_init, t_lims, num_points=4, num_coeff_per_dim=4)



