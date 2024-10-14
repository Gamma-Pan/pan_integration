import torch
from torch import tensor, nn, linalg, cos, sin
from torch.nn import functional as F
from torch.func import jacrev
from pan_integration.utils.plotting import wait, DimPlotter

from scipy.optimize import linprog
from torchdiffeq import odeint


import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("TkAgg")
# torch.set_default_dtype(torch.float64)

torch.manual_seed(23)
from torch.linalg import inv
from pan_integration.utils.plotting import wait
from pan_integration.core.solvers import T_grid, DT_grid, PanSolver


torch.manual_seed(68)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


if __name__ == "__main__":

    a = 1000

    def f(t, y):
        return -a * y + a * sin(t)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    y_init = tensor([0.0])
    y_init = y_init
    t_lims = tensor([-1.0, 1.0])
    f_init = f(t_lims[0], y_init)

    NUM_COEFF = 10
    NUM_POINTS = 50

    t_span = torch.linspace(*t_lims, NUM_POINTS)

    N = NUM_COEFF

    k = torch.arange(0, NUM_POINTS)

    ts = torch.linspace(*t_lims, NUM_POINTS)
    # ts = -cos(torch.pi * k / (N - 1))

    Phi = T_grid(ts, NUM_COEFF).mT
    DPhi = DT_grid(ts, NUM_COEFF).mT

    Phi_c = DPhi[:, 2:] - DPhi[[0], 2:]
    Phi_u = Phi[[0], 2:] + DPhi[[0], 2:] + DPhi[[0], 2:] * ts[:, None] - Phi[:, 2:]
    d = y_init + (1 + ts[:, None]) * f_init

    # sol_cheby = d - Phi_u @ B_ch.reshape(NUM_COFF-2, 1)
    # print(sol_cheby.shape)

    # b = B_ch.reshape(NUM_COFF-2, 1)
    b = torch.rand(NUM_COEFF - 2, 1)

    # for i in range(3):
    #     b = torch.linalg.lstsq(Phi_c, f(0, d - Phi_u @ b) - f_init).solution

    prev_b = b
    for i in range(10):
        yk = d - Phi_u @ b
        fk = f(ts.unsqueeze(1), yk)
        J = jacrev(f, argnums=1)(ts, yk)[:, 0, :, 0]
        A_q = Phi_c + J @ Phi_u
        con = f_init + Phi_c @ b - fk
        c = torch.cat([torch.zeros_like(b), torch.tensor([[1.0]])], dim=0)
        A = torch.hstack([torch.vstack([A_q, -A_q]), -torch.ones(2 * (NUM_POINTS), 1)])

        res = linprog(
            c=c,
            A_ub=A,
            b_ub=torch.vstack([-con, con]),
            bounds=[None, None],
            method="highs",
        )

        b = b + tensor(res.x, dtype=torch.float)[:-1, None]

        delta = res.x[-1]
        print(i, delta)
        if torch.norm(prev_b - b) < 1e-8:
            break
        prev_v = b

    print(f"b_nlp \n {b.reshape(-1,1)}")

    sol_nlp = d - Phi_u @ b
    sol_true = odeint(f, y_init, ts)

    plt.plot(ts, sol_true.squeeze(), "g")
    plt.plot(ts, sol_nlp.squeeze(), "r")
    # plt.plot(
    #     ts,
    #     torch.log10(
    #         abs( sol_true.squeeze() - sol_nlp.squeeze())
    #     ),
    #     "r--",
    # )
    # plt.ylim(1, 4)
    plt.show()
