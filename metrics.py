import torch
from torch import tensor, tanh
from pan_integration.solvers import pan_int
from pan_integration.utils import plotting
from pan_integration.solvers.pan_integration import _B_init_cond, _cheb_phis
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import norm

# np.set_printoptions(suppress=True, precision=10)


def spiral(batch):
    a = 0.5
    P = tensor([[-a, 1.0], [-1.0, -a]])[None]
    xy = batch[..., None]  # add trailing dim for matrix vector muls
    # derivative = tanh(-P @ sin(0.5 * P @ tanh(P @ xy)))
    # derivative = P @ xy
    derivative = tanh(P @ xy)
    return torch.squeeze(derivative)


if __name__ == "__main__":
    metrics = {}

    t_lims = [0, 6]
    y_init = tensor([-1, -1], dtype=torch.float)

    def f(t, x):
        return spiral(tensor(x, dtype=torch.float)).numpy()

    solution = solve_ivp(f, t_lims, y_init, method="RK45", max_step=1e-3)
    yT = solution.y[:, -1]
    print(f"truish: \t {solution.y[:,-1]} \t {solution.nfev}")

    fig, ax = plt.subplots()
    errors = [[],[],[]]
    for i, step in enumerate((1,2,3)):
        for numc in range(10,40):
            pan_y, pan_nfe = pan_int(
                spiral,
                y_init,
                t_lims,
                num_coeff_per_dim_newton=numc,
                etol=1e-8,
                atol=1e-12,
                num_points=100,
                step=step,
                coarse_steps=7,
                callback=None,
            )
            err = norm(yT - pan_y[-1,:].numpy())
            print(f"pan ({numc}, {step}) \t {err:.10} \t {pan_nfe}")
            errors[i].append(err)

        plt.yscale('log')
        ax.plot(list(range(10,40)), errors[i], label=str(step))

    ax.legend()
    plt.show()

    methods = ["RK45", "DOP853", "BDF", "LSODA"]
    for method in methods:
        solution = solve_ivp(f, t_lims, y_init, method=method, rtol=1e-13, atol=1e-8)
        print(
            f"{method} \t {norm(yT - solution.y[:,-1]):.10} \t {solution.nfev} \t {( solution.nfev/ pan_nfe) * 100:3.0f}%"
       )
