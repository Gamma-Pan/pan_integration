import torch
from torch import tensor, cos, sin, hstack, vstack, norm, tanh, cosh
from torch.linalg import inv
from torch import pi as PI
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pan_integration.solvers.pan_integration import (
    T_grid,
    U_grid,
    _cheb_phis,
    _coarse_euler_init,
)
from pan_integration.utils.plotting import wait
from pan_integration.solvers import pan_int


num_points = 100
num_coeff = 40
t_lims = tensor([0, 5])
y0 = torch.tensor([0.3,0.2])


def lorentz(batch, sigma=10, rho=28, beta=8 / 3):
    x, y, z = torch.unbind(batch[..., None], 1)
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return torch.stack((dxdt, dydt, dzdt), dim=1).squeeze()


def f2(batch):
    a = 0.5
    P = tensor([[-a, 1.0], [-1.0, -a]])[None]
    xy = batch[..., None]  # add trailing dim for matrix vector muls
    derivative = tanh(P @ xy)
    return torch.squeeze(derivative)


def f(x):
    return  cos(sin(torch.pi*x))**2


def poly_approx_ivp_ls(f, filename):
    f0 = f(y0[None])

    t_eval = torch.linspace(*t_lims, num_points)
    ode_sol = solve_ivp(
        lambda t, x: f(tensor(x[None], dtype=torch.float)),
        t_lims,
        y0,
        t_eval=t_eval,
        method="LSODA",
        atol=1e-9,
        rtol=1e-13,
    )
    y_solver = ode_sol.y

    fig, ax = plt.subplots()
    ax.plot(*y_solver, label="solver")
    art, = ax.plot([],[])
    # ax.legend()

    Phi, DPhi = _cheb_phis(num_points, num_coeff, t_lims)
    inv0 = inv(vstack((Phi[0, [0, 1]], DPhi[0, [0, 1]])))
    Phi_a = DPhi[:, [0, 1]] @ inv0 @ vstack((y0, f0))
    Phi_b = (
        -DPhi[:, [0, 1]] @ inv0 @ vstack((Phi[[0], 2:], DPhi[[0], 2:])) + DPhi[:, 2:]
    )
    l = lambda b: inv0 @ (vstack((y0, f0)) - vstack((Phi[[0], 2:], DPhi[[0], 2:])) @ b)

    num_solver_steps = 5
    b = _coarse_euler_init(f, y0, num_solver_steps, t_lims, num_coeff)[:, None].reshape(2,num_coeff-2).T

    Q = inv(Phi_b.T @ Phi_b)
    bs = [b]
    errors = []
    for i in range(50):
        b_prev = b
        b = Q @ (Phi_b.T @ f(Phi @ vstack((l(b), b))) - Phi_b.T @ Phi_a)
        bs.append(b)

        y_hat = Phi @ vstack((l(b), b))
        error = norm(y_hat - y_solver.T)

        errors.append(error)

        if norm(b - b_prev) < 1e-10:
            break

    b = bs[-1]
    y_hat = Phi @ vstack((l(b), b))
    yT = y_solver[0, -1]

    # print(f"rk45 took {ode_sol.nfev} nfes,\t y(T) = {yT}")
    #
    # print(
    #     f"ls took {len(bs)+num_solver_steps} nfes,\t y(T) = {y_hat[-1,0].numpy()},\t error={yT - y_hat[-1,0]:e}"
    # )

    sol_pan, nfe = pan_int(f, y0, t_lims.tolist(), num_coeff, num_points , coarse_steps=num_solver_steps, etol_newton=1e-10)

    # print(
    #     f"pan took {nfe} nfes,\t y(T) = {sol_pan[-1,0].numpy()},\t  error={yT - sol_pan[-1,0]:e}"
    # )

    ax.plot(*sol_pan.T, label="pan")
    ax.plot(*y_hat.T, label="ls")
    ax.legend()
    plt.show()

    def update(frame):
        b = bs[frame]
        y_hat = Phi @ vstack((l(b), b))
        error = DPhi @ vstack((l(b), b)) - f(DPhi @ vstack((l(b), b)))
        art.set_data(t_eval, y_hat[:, 0])

        ax.set_title(f"#coeff = {num_coeff}   iter: {frame}   MSE={norm(error)}")
        return art

    from matplotlib import animation

    # ani = animation.FuncAnimation(fig=fig, func=update, frames=len(bs), interval=100)
    # ani.save(filename, writer="pillow")
    # plt.show()

    print("fin")


if __name__ == "__main__":
    poly_approx_ivp_ls(f2, "remez_like.gif")
