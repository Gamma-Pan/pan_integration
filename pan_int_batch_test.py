from pan_integration.solvers.pan_integration import (
    lst_sq_solver,
    _coarse_euler_init,
    pan_int,
)
from pan_integration.optim.minimize import newton
import torch


def f(batch):
    A = torch.arange(10, dtype=torch.float).reshape(5, 2)[None, ...]
    return (A @ batch[:, :, None])[...,0]


if __name__ == '__main__':

    y_init = torch.rand(4, 2, dtype=torch.float)
    y = f(y_init)

