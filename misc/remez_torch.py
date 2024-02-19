from typing import Iterable
import torch
from torch import (
    zeros_like,
    nn,
    sin,
    tensor,
    sign,
    diff,
    abs,
)
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

matplotlib.use("TkAgg")

PI = torch.pi


class NeuralField(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        t, z = torch.split(x, 1, -1)

        y = 1.5 * sin(PI * t * z * 5.6) + t * 0.5
        return y


vector_field = NeuralField()


def ode(t, y):
    with torch.no_grad():
        t = torch.tensor([t])
        y = torch.tensor(y)
        out = vector_field(torch.stack((t, y), dim=1))
        return out[0, 0]


def plot_vf(ax):
    # plot vector field
    num_points = 40
    grid_x = torch.linspace(0, 1, num_points)
    grid_y = torch.linspace(0, 1, num_points)
    mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
    mesh = torch.dstack((mesh_x, mesh_y))
    batch = mesh.reshape((num_points * num_points, 2))

    derivatives = vector_field(batch).detach()
    derivatives_y = derivatives.reshape(num_points, num_points)
    derivatives_x = torch.ones(num_points, num_points)

    # solve ode numerically
    sol = solve_ivp(ode, [0, 1], [0.4], method="LSODA")

    ax.quiver(mesh_x, mesh_y, derivatives_x, derivatives_y)
    ax.plot(sol.t, sol.y[0, :], "r-o", markersize=4)



def Phi_t(t: torch.Tensor):
    # important for broadcasting
    t = t.reshape(-1, 1)
    t_size = t.shape[0]
    exponents = torch.arange(2, t_size - 1)[None, :]
    powers = torch.pow(t, exponents)
    alter_ones = torch.ones(t_size, 1)
    alter_ones[1::2, 0] = -1
    ones = torch.ones((t_size, 1))
    mat = torch.hstack((ones, t, powers, alter_ones))
    assert mat.shape[0] == mat.shape[1]
    return mat


def remez(true_f: torch.Tensor, num_params: int = 9):
    init_coeff = torch.rand(num_params)


def f_generator():
    p = torch.randn(5) * 3
    def f(t):
        return p[0] * t * sin(PI * p[1] * t * t) * p[2] + p[3] * t + p[4]

    return f


def keep_max_between_roots(error: torch.Tensor) -> torch.Tensor:
    all_extrema_idx = diff(sign(diff(error))).nonzero() + 1
    all_extrema_idx = torch.cat(
        [tensor([[0]]), all_extrema_idx, tensor([[error.shape[0] - 1]])]
    ).squeeze()
    crossings_idx = diff(sign(error[all_extrema_idx]), dim=0).nonzero().squeeze() + 1

    area_tuples = torch.tensor_split(all_extrema_idx, crossings_idx)
    area_max_crossings_idx = torch.stack(
        [torch.argmax(abs(error[x])) for x in area_tuples]
    )

    crossings_idx_pz = torch.cat((tensor([0]), crossings_idx))
    all_alter_extrema = all_extrema_idx[area_max_crossings_idx + crossings_idx_pz]

    return all_alter_extrema


def choose_new_control(error: torch.Tensor, num_points: int):
    abs_error = abs(error)
    all_extrema_idx = keep_max_between_roots(error)
    indices_init = all_extrema_idx.tolist()

    def _recursive_delete(indices: list):
        size = len(indices)
        if size == num_points:
            return indices
        elif size == num_points+1:
            if abs(error[indices[0]]) > abs(error[indices[-1]]):
                return indices[:-1]
            else:
                return indices[1:]
        else:
            # get the index of the smallest extrema
            smallest_idx = torch.argmin(abs_error[indices])

            # also get its smallest neighbour
            if (
                abs_error[indices[(smallest_idx - 1) % size]]
                < abs_error[indices[(smallest_idx + 1) % size]]
            ):
                smallest_neighbour = (smallest_idx - 1) % size
            else:
                smallest_neighbour = (smallest_idx + 1) % size

            # remove smallest and smallest neighbour
            smallest_idx = indices[smallest_idx]
            smallest_neighbour = indices[smallest_neighbour]
            indices.remove(smallest_idx)
            indices.remove(smallest_neighbour)

            return _recursive_delete(indices)

    return _recursive_delete(indices_init)

plotting = True

if __name__ == "__main__":
    f = f_generator()
    t_all = torch.linspace(0, 1, 100)
    f_all = f(t_all)

    # initial control points
    num_coeffs = 9
    t_control = torch.linspace(0.1, 0.9, num_coeffs + 1)

    for i in range(10):
        print("---->" + str(i) + "<----")
        f_control = f(t_control)
        Power_Mat = Phi_t(t_control)

        # find coefficients
        solution = torch.linalg.solve(Power_Mat, f_control)
        coeffs = solution[:-1]  # last is the residual

        # reconstructions
        exponents = torch.arange(0, num_coeffs)[None]
        reconstruction = torch.sum(
            torch.pow(t_all.reshape(-1, 1), exponents) * coeffs, dim=1
        )

        delta = torch.sum(
            torch.pow(t_control.reshape(-1, 1), exponents) * coeffs, dim=1
        )

        # error
        error = f_all - reconstruction

        control_idx = choose_new_control(error, num_coeffs+1)

        if plotting:
            fig, (ax, ax_delta) = plt.subplots(1, 2, figsize=(12, 8))
            plt.setp(ax, facecolor="#eee", xlim=[-0.1, 1.1])
            ax.grid(color="#aaa", linewidth=0.7)
            ax.plot([-10, 10], [0, 0], "#777", linewidth=1.2)
            ax.plot(t_all, f_all, "forestgreen")

            ax.plot(t_all, reconstruction, "rebeccapurple", label="minimax approx")
            ax.plot(
                torch.stack((t_control, t_control)),
                torch.stack((f_control, delta)),
                "r-o",
                markersize=4,
            )

            plt.setp(ax_delta, facecolor="#eee", xlim=[-0.1, 1.1])
            ax_delta.grid(color="#aaa", linewidth=0.7)
            ax_delta.plot([-10, 10], [0, 0], "#777", linewidth=1.2)
            ax_delta.plot(t_all, error, color="orange", label="delta", markersize=3)
            error_init = f_control - delta
            ax_delta.plot(
                torch.vstack((t_control.view(-1), t_control.view(-1))),
                torch.vstack((torch.zeros(num_coeffs + 1), f_control - delta)),
                "r-o",
                markersize=4,
            )

            while True:
                if plt.waitforbuttonpress():
                    #plt.close(fig)
                    break

            t_control = t_all[control_idx]

