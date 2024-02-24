import torch
from torch import tensor, squeeze, sin
from pan_integration.utils.plotting import VfPlotter, wait


def f(batch, a=0.2):
    P = tensor([[-a, 1.0], [-1.0, -a]])[None, None]

    xy = batch[..., None]
    derivative =  P @ xy
    return squeeze(derivative)


if __name__ == "__main__":
    y_init = tensor([-0.7, -0.7])

    ax_kwargs = {"xlim": (-1, 1), "ylim": (-1, 1)}
    plotter = VfPlotter(f, ax_kwargs=ax_kwargs)
    plotter.solve_ivp(y_init, {"method": "RK45"}, {"color":"red"})
    plotter.solve_ivp(y_init, {"method": "LSODA"}, {"color":"blue"})

    wait()
