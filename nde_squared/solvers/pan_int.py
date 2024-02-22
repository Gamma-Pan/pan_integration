import torch
from torch import tensor, Tensor
from ..optim import newton

def pan_int(f, y_init, grid=None, etol=1e-3, Phi=None, DPhi=None)->Tensor:

