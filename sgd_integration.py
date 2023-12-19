import torch
from torch import nn, tensor, cos, pi, sin, abs, zeros, sum, arctan, sqrt
from torch.optim import Adam, SGD
from torch.func import hessian, jacrev, jacfwd
from torch.linalg import inv
from scipy.integrate import solve_ivp
import numpy as np
from functools import partial
from nde_squared.utils.plotting import VfPlotter
from nde_squared.optim.line_search import line_search
import matplotlib.pyplot as plt
from typing import Callable


def f_generator():
    W1 = torch.randn(1, 2, 2)
    b1 = torch.rand(1, 2, 1)
    W2 = torch.randn(1, 10, 2)
    b2 = torch.rand(1, 10, 1)
    W3 = torch.randn(1, 2, 10)
    b3 = torch.rand(1, 2, 1)
    tanh = torch.nn.Tanh()

    def f(y):
        y = y.reshape((-1, 2, 1))
        l1 = tanh(W1 @ y + b1)
        l2 = tanh((W2 @ l1 + b2))
        out = sin(0.7 *2 * torch.pi * W3 @ l2 + b3)
        return out.reshape((-1, 2))

    return f


def naive_newton():
    raise NotImplementedError


def objective_f(beta, f, y_init, num_points=100, vf_plotter=None):
    """
    A Rn-> R function that we want to minimize wrt. beta.

    :param beta: parameters vector
    :param f: dy/dt = f(y(t))
    :param y_init: initial value
    :param num_points: the number of point on which error is calculated
    :return: scalar error based on criterion
    """

    # make 1d parameter vector into matrix, transpose(?)
    beta = beta.reshape(-1, torch.numel(y_init))
    num_coeff = beta.shape[0]

    b = torch.cat([f(y_init), beta], dim=0)

    t = torch.linspace(0, 1, num_points + 1)[:, None]  # column vector
    exponents = torch.arange(1, num_coeff + 2)[None:]  # row vector

    Phi_y = torch.pow(t, exponents)
    Phi_dy = torch.pow(t, exponents - 1) * exponents  # broadcasting

    y_approx = y_init + Phi_y @ b

    dy_approx = Phi_dy @ b
    f_y_approx = f(y_approx)

    residuals = torch.pow(dy_approx - f_y_approx, 2)
    error = residuals.sum() / (num_points * num_coeff * torch.numel(y_init))

    return error


def coarse_euler(y_init, f, steps):
    t_eval = torch.linspace(0, 1, steps + 1)[1:, None]
    step = 1 / steps
    sol = []
    cur_y = y_init
    for t in t_eval:
        sol.append(cur_y)
        next_y = cur_y + step * f(cur_y)
        cur_y = next_y

    return torch.cat(sol, dim=0)


if __name__ == "__main__":
    f = f_generator()
    y_init = torch.rand(1, 2)
    f_init = f(y_init)
    vf_plot = VfPlotter(f, y_init)
    dims = torch.numel(y_init)

    num_coeff = 10  # total
    obj_f = partial(objective_f, f=f, y_init=y_init, vf_plotter=vf_plot)
    jacobian = jacrev(obj_f)
    hessian = jacfwd(jacobian)

    steps = int(num_coeff / dims + 2)
    traj_euler = coarse_euler(y_init, f, steps)

    print("Euler solution")
    vf_plot.approximation(torch.cat([y_init, traj_euler]))

    # solve a least squares problem to init betas near euler solution
    t_euler = torch.linspace(0, 1, steps)[:, None]  # column vector
    exponents = torch.arange(1, steps)[None, :]  # row vector
    Phi_y = torch.pow(t_euler, exponents)
    b_lstsq = torch.linalg.lstsq(Phi_y, traj_euler - y_init).solution

    t_plot = torch.linspace(0, 1, 100)[:, None]  # column vector
    Phi_y = torch.pow(t_plot, exponents)
    Phi_dy = torch.pow(t_plot, exponents - 1) * exponents  # broadcasting

    print("Least squares")
    vf_plot.approximation(
        (Phi_y @ b_lstsq) + y_init, Phi_y @ b_lstsq, f
    )

    b = b_lstsq[1:, :].reshape(-1).detach()

    for i in range(100):
        Jb = jacobian(b)
        Hb = hessian(b)

        Hnp = Hb.numpy()
        Hb_inv = torch.linalg.inv(Hb)

        p = -Hb_inv @ Jb

        b = line_search(obj_f, jacobian, b, p)

        with torch.no_grad():
            if i == 0:
                print("Starting Newton")

            beta = b.reshape(-1, dims)
            beta = torch.cat([f_init, beta], dim=0)

            y_approx = y_init + Phi_y @ beta
            vf_plot.approximation(y_approx, Phi_dy@beta, f=f)
