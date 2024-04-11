import torch
from torch.func import vmap
from torch import tensor, nn
from torch.masked import masked_tensor

import matplotlib.pyplot as plt
import matplotlib as mpl

from pan_integration.optim import newton
from pan_integration.optim.factor import mod_chol

mpl.use("TkAgg")


class Rosenbrock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p):
        x = p[..., 0]
        y = p[..., 1]
        a = 1
        b = 100

        return (a - x) ** 2 + b * (y - x**2) ** 2


rosenbrock = Rosenbrock()

def plot_rosen():
    xs = torch.linspace(-2, 2, 100)
    ys = torch.linspace(-1, 3, 100)

    Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    Ps = torch.stack((Xs, Ys), dim=-1).reshape(-1, 2)
    Zs = rosenbrock(Ps).reshape(100, 100)
    ax.plot_surface(Xs, Ys, Zs, cmap="viridis")
    return ax

if __name__ == "__main__":

    point_init = tensor([[-1.2, 0.4]])
    # point_init = tensor([[-1.7, -0.7], [ -1.2, 0.4 ]])

    point_min = newton(rosenbrock, point_init, etol=1e-10)
    print(point_min)
