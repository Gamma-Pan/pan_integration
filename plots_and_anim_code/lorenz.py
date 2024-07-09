import torch
from torch import cat, stack, tensor
import matplotlib as mpl
mpl.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pan_integration.core.solvers import PanSolver
import torchdyn
from torchdyn.numerics import odeint


def Lorenz(t, input, sigma=10, rho=28, beta=8 / 3):
    x = input[..., 0]
    y = input[..., 1]
    z = input[..., 2]

    dx = sigma * (y - x)
    dy = x*(rho - z) - y
    dz = x * y - beta * z

    return torch.stack([dx, dy, dz], dim=-1)

# %%
y_init = tensor([0, 1, 1.05])

t_tsit, traj_tsit = odeint(
    Lorenz, y_init, solver="dopri5", atol=1e-9,rtol=1e-9, t_span=torch.linspace(0, 100, 100)
)

fig  = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection="3d" )
ax2 = fig.add_subplot(122, projection="3d" )

ax1.plot(*traj_tsit[::6].unbind(0), color='blue')
ax1.set_xlim(-10,10)
ax1.set_ylim(-10,10)
ax1.set_zlim(-10,10)
plt.show()


























