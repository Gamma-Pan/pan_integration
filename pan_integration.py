import torch
from torch import nn, tensor, cos, pi, sin, abs, zeros, sum, arctan, sqrt
from torch.optim import Adam, SGD
from torch.func import hessian, jacrev, jacfwd
from torch.linalg import inv
from scipy.integrate import solve_ivp
import numpy as np
from functools import partial
from pan_integration.utils.plotting import VfPlotter
from pan_integration.optim.line_search import line_search
from pan_integration.optim.factor import mod_chol
import matplotlib.pyplot as plt
from typing import Callable

torch.manual_seed(435)
# BAD : 845:26
# GOOD : 435:26, 1423:26
# GOODish :123:14, 984:18, 434:14, 656:26

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
        out = sin(1.7 * 2 * torch.pi * W3 @ l2 + b3)
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

    global NFE
    NFE = NFE+1

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
    y = torch.rand(1, 2)

    # ivp_kwargs = dict({"method": "RK45"} )
    # vf_plot = VfPlotter(f, y, ivp_kwargs=ivp_kwargs,show=True)
    vf_plot = VfPlotter(f, y, show=True)
    # vf_plot.ivp(y)

    dims = torch.numel(y)
    num_coeff = 26  # total
    obj_f = partial(objective_f, f=f, y_init=y, vf_plotter=vf_plot)

    grad = jacrev(obj_f)
    hessian = jacfwd(grad)

    step = 1
    num_steps = int(1 / step)

    exponents = torch.arange(1, int(num_coeff / dims + 2))[None, :]  # row vector

    NFE = 0
    for idx in range(num_steps):
        print(f"from y = {y} \n")

        # linear step along descent direction
        t_l = torch.linspace(0, step, int(num_coeff / dims + 1))[:, None]
        # y_l = y + step * t_l * f(y)

        Phi_y = torch.pow(t_l, exponents)
        Phi_dy = torch.pow(t_l, exponents - 1) * exponents  # broadcasting

        traj_euler = coarse_euler(y, f, int(num_coeff / dims + 1))

        # solve a least squares problem to init b for this step
        b_lstsq = torch.linalg.lstsq(Phi_y, traj_euler - y).solution
        # b_lstsq = torch.linalg.lstsq(Phi_y, y_l - y).solution

        tol = 1e-4
        b = b_lstsq[1:, :].reshape(-1).detach()
        for i in range(50):
            # minimize error for n points along line
            Db = grad(b)[:,None]
            print(torch.norm(Db))
            if (torch.norm(Db)) < tol:
                y = y_approx[-1, :]
                break

            Hb = hessian(b)
            # make hessian positive definite
            LD = mod_chol(Hb)

            pivots = torch.arange(1, num_coeff+1)
            d = -torch.linalg.ldl_solve(LD, pivots, Db)

            b = line_search(obj_f, grad, b, d[:,0])

            with torch.no_grad():
                beta = b.reshape(-1, dims)
                beta = torch.cat([f(y), beta], dim=0)

                y_approx = y + Phi_y @ beta

    print(f"Number of functions evaluations: {NFE}")

    plt.show()
