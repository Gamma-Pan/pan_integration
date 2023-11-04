from typing import Iterable
import torch
from torch import nn, sin, sum, pow
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# CONSTANSTS
LIMITS = [0, 1]
PAD = 0.1
PI = torch.pi


# the function to approximate
def true_f(t: torch.Tensor, a: torch.Tensor = None, b: torch.Tensor = None):
    if len(t.size()) == 1:
        t = torch.unsqueeze(t, dim=-1)
    assert b.size() == torch.Size([1])
    exp = torch.arange(1, a.size()[0] + 1)
    out = sum(a * pow(t, exp), dim=1) + b
    return out


t = torch.arange(0, 1, 0.01)
a = torch.tensor([0.5, -0.3, -0.1])
b = torch.tensor([0.5])
gt = true_f(t, a, b)

fig, ax = plt.subplots()
ax.grid(True, color="0.85")
ax.plot(t, gt)
ax.plot(0, b, markerfacecolor="red", marker="o", markersize=4)
plt.show()


# Remez exchange algorithm
