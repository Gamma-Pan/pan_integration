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
DIMS = 1
NUM_COFF = 10


class NN(nn.Module):
    def __init__(self, std=5.0):
        super().__init__()
        self.w1 = torch.nn.Linear(DIMS, 10)
        torch.nn.init.normal_(self.w1.weight, std=std)
        self.w2 = torch.nn.Linear(10, 10)
        torch.nn.init.normal_(self.w2.weight, std=std)
        self.w3 = torch.nn.Linear(10, DIMS)
        torch.nn.init.normal_(self.w3.weight, std=std)
        self.A = torch.tensor([[-0.9, -2.0], [1.5, -1]], device=device)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        y = torch.tanh(self.w1(y))
        y = F.softplus(self.w2(y))
        y = torch.cos(self.w3(y))
        return y


alpha = 3


class Fun(nn.Module):
    def __init__(self, c=0.1):
        super().__init__()
        self.nfe = 0
        self.c = c

    def forward(self, t, y):
        self.nfe += 1
        return cos(y) + sin(y)


if __name__ == "__main__":

    # f = NN(std=3).to(device)
    f = Fun(c=0.5).to(device)
    for param in f.parameters():
        param.requires_grad_(False)

    # y_init = torch.tensor([[0.123]], device=device)
    y_init = torch.randn((1, DIMS), device=device)
    t_lims = tensor([-1.0, 1.0], device=device)
    t_span = torch.linspace(*t_lims, 2, device=device)

    # plotter = DimPlotter(
    #     f,
    #     [
    #         [0, 0],
    #         # [0, 1],
    #         # [0, 2],
    #         # [0, 3],
    #         # [0, 4],
    #         # [0, 5],
    #         # [0, 6],
    #         # [0, 7],
    #         # [0, 8],
    #     ],
    # )

    # tsit_sol = plotter.solve_ivp(
    #     t_span,
    #     y_init,
    #     ivp_kwargs=dict(solver="dopri5", atol=1e-4, rtol=1e-4),
    #     plot_kwargs=dict(color="blue", alpha=0.1),
    # )

    # exact_sol = plotter.solve_ivp(
    #     t_span,
    #     y_init,
    #     plot_kwargs=dict(color="green", alpha=0.1),
    # )

    # plotter.axes.plot(torch.linspace(*t_lims,1000), (torch.exp(torch.linspace(*t_lims,1000))*torch.e*y_init)[0], 'lime')

    # dop_err = torch.norm(exact_sol[-1] - tsit_sol[-1])

    # print(f"dopri5 sol {torch.norm(exact_sol[-1] - tsit_sol[-1])}")
    # wait()

    # f.nfe = 0
    # approx_text = plotter.fig.text(x=0.6, y=0.05, s=f"nfe = {f.nfe}", color="black")

    # writer = FFMpegFileWriter(fps=4, metadata=dict(title="ode sol"))
    # writer.setup(plotter.fig, "output_arr.mp4", dpi=300)
    # writer.frame_format = "png"

    # def callback(i, t_lims, y_init, f_init, B, **kwargs):
    #     approx = plotter.approx(
    #         tensor(t_lims),
    #         B,
    #         num_arrows=0,
    #         num_points=101,
    #         marker=None,
    #         markersize=2.5,
    #         alpha=0.70,
    #         color="red",
    #     )
    #
    #     approx_text.set_text(f"nfe = {f.nfe} ")
    #     plotter.fig.canvas.flush_events()
    #     plotter.fig.canvas.draw()

    # solver = PanSolver(
    #     num_coeff_per_dim=NUM_COFF,
    #     callback=callback,
    #     device=device,
    #     tol=1e-5,
    #     max_iters=100,
    #     gamma=1,
    # )

    t_span = torch.linspace(*t_lims, 2, device=device)

    # approx, _, B_ch = solver.solve(
    #     f,
    #     t_span,
    #     y_init,
    #     gamma=0.98
    # )

    # print(f"b_cheby \n {B_ch.reshape(-1,1)}")

    # writer.finish()
    # print(f"pan sol {torch.norm(approx[-1] - exact_sol[-1])}")
    # plt.show()

    #########################################################################

    def f(t, y):
        return torch.sqrt(torch.abs(cos(y) + sin(y)))

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    y_init = tensor([0.0])
    y_init = y_init
    f_init = f(0.0, y_init)
    t_lims = [-1.0, 1.0]

    NUM_COEFF = 20
    NUM_POINTS = 50

    t_span = torch.linspace(*t_lims, NUM_POINTS)

    N = NUM_COEFF

    k = torch.arange(0, NUM_POINTS)

    # ts = torch.linspace(*t_lims, NUM_POINTS)
    ts = -cos(torch.pi * k / (N - 1))

    Phi = T_grid(ts, NUM_COEFF).mT
    DPhi = DT_grid(ts, NUM_COEFF).mT

    Phi_c = DPhi[:, 2:] - DPhi[[0], 2:]
    Phi_u = Phi[[0], 2:] + DPhi[[0], 2:] + DPhi[[0], 2:] * ts[:, None] - Phi[:, 2:]
    d = y_init + (1 + ts[:, None]) * f_init

    # sol_cheby = d - Phi_u @ B_ch.reshape(NUM_COFF-2, 1)
    # print(sol_cheby.shape)

    # b = B_ch.reshape(NUM_COFF-2, 1)
    b = torch.rand(NUM_COEFF-2,1)

    # for i in range(3):
    #     b = torch.linalg.lstsq(Phi_c, f(0, d - Phi_u @ b) - f_init).solution

    prev_b = b
    for i in range(10):
        yk = d - Phi_u @ b
        fk = f(0, yk)
        J = jacrev(f, argnums=1)(0, yk)[:, 0, :, 0]
        A_q = Phi_c + J @ Phi_u
        # A_q = Phi_c #+ J @ Phi_u
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
        if i>5 or torch.norm(prev_b - b) < 1e-8:
            break
        prev_v = b

    print(f"b_nlp \n {b.reshape(-1,1)}")

    sol_nlp = d - Phi_u @ b
    # print(sol_nlp)

    print(torch.norm(b,2))
    # print(torch.norm(b - B_ch.mT))

    # print(f" {torch.max(torch.abs((sol_nlp - sol_cheby) / sol_nlp)) * 100 :.3} %")

    sol_true = odeint(f, y_init, ts)

    # plt.plot(
    #     ts,
    #     torch.log10(
    #         abs(sol_true.squeeze() - sol_cheby.squeeze())
    #     ),
    #     "g--",
    # )
    plt.plot(
        ts,
        torch.log10(
            abs( sol_true.squeeze() - sol_nlp.squeeze())
        ),
        "r--",
    )
    # plt.ylim(1, 4)
    plt.show()
