import torch
from pan_integration.core.solvers import PanSolver2
from torch import nn
from torch.nn import functional as F

DIM = 10
factor = 1

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


