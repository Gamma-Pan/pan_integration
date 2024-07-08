import numpy as np
from numpy import linspace, meshgrid, pi, sin, cos
from scipy.integrate import solve_ivp

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg")
mpl.rc("axes", edgecolor="55555588")

# copy pasted code
# mpl.rcParams.update(
#     {
#         "pgf.texsystem": "pdflatex",
#         "font.family": "serif",
#         "text.usetex": True,
#         "pgf.rcfonts": False,
#     }
# )
plt.close("all")


def grad_f(t, y):
    return np.array([t + 1, 2 * sin(pi * t * y)])


t_lims = [0.0, 1.0]
t_grid_def = 20
y_lims = [-0.6, 0.6]
y_grid_def = 20

cm = 1 / 2.54
# fig_ode, ax = plt.subplots(2, 2, layout="tight", figsize=(12 * cm, 12 * cm))
# fig_ode.suptitle("ResNet as discretisation of ODEs")

init_points = np.linspace(-0.1, 0.1, 5)
init_points = np.vstack((np.zeros(init_points.shape[0]), init_points)).T


def resnet_states(
    ax,
    count,
    plot_kwargs=None,
    discrete=True,
):
    if plot_kwargs is None:
        plot_kwargs = {}

    for idx, point in enumerate(init_points):
        sol = solve_ivp(
            lambda t, y: grad_f(y[0], y[1]),
            t_lims,
            point,
            "LSODA",
            t_eval=np.linspace(t_lims[0], t_lims[1], count),
        )

        # if idx == 4:
        #     ax.text(sol.y[0][0] + 0.01, sol.y[1][0] + 0.05, "test")

        solution = sol.y
        if discrete:
            ax.set_xticks(ticks=solution[0], labels=[])
            ax.set_xlabel("depth")
        else:
            ax.set_xlabel("time")
            ax.set_xticks(ticks=[])

        ax.plot(solution[0], solution[1], zorder=100, **plot_kwargs)

    ax.set_yticks(ticks=[])

    ax.set_xlim(*t_lims)
    ax.set_ylim(*y_lims)

    ax.set_ylabel("state")


t_mesh = linspace(t_lims[0], t_lims[1], t_grid_def)
y_mesh = linspace(y_lims[0], y_lims[1], y_grid_def)
T_mesh, Y_mesh = meshgrid(t_mesh, y_mesh, indexing="xy")

dT, dY = grad_f(T_mesh, Y_mesh)


def ode_streams(ax, opacity, stream_kwargs=None):
    if stream_kwargs is None:
        stream_kwargs = {}

    stream_kwargs["color"] = "#555555" + opacity

    ax.streamplot(
        T_mesh,
        Y_mesh,
        dT,
        dY,
        **stream_kwargs,
        # broken_streamlines=False,
        density=0.7,
        arrowsize=0.6,
        linewidth=0.6
    )


fig, ax = plt.subplots()
# draw resnet states
resnet_states(ax, 7, {"marker": "o"})
ode_streams(ax, "10")
plt.savefig("resnet_to_node_plot_1.png")

fig, ax = plt.subplots()
resnet_states(ax, 18, {"marker": "o"})
ode_streams(ax, "20")
plt.savefig("resnet_to_node_plot_2.png")

fig, ax = plt.subplots()
resnet_states(ax, 50, {"marker": "o", "linestyle": "-", "markersize": 4})
ode_streams(ax, "30")
plt.savefig("resnet_to_node_plot_3.png")

fig, ax = plt.subplots()
resnet_states(
    ax,
    100,
    {
        "linestyle": "-",
        "linewidth": 3,
    },
    discrete=False,
)
ode_streams(ax, "80")
plt.savefig("resnet_to_node_plot_4.png")

# plt.show()
# plt.savefig("resnet_to_node_plot.pgf")
