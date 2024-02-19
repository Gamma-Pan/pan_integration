import torch
from torch import tensor, abs, norm, squeeze
import matplotlib.pyplot as plt
import matplotlib as mpl
from nde_squared.optim.line_search import find_step_length
from torch.func import jacfwd, jacrev, hessian

mpl.use("TkAgg")


def wait():
    while True:
        if plt.waitforbuttonpress():
            break


def quadratic(batch) -> tensor:
    batch = batch[...,None]
    A = tensor([[4., 1.], [1., 2.]])
    b = tensor([[0.5, 0.5]])

    heights = batch.transpose(-2, -1) @ A @ batch + b @ batch

    return squeeze(heights)


def rosenbrock(batch) -> tensor:
    a = 1
    b = 100

    x = batch[..., 0]
    y = batch[..., 1]

    return (a - x) ** 2 + b * ((y-x) ** 2) ** 2


if __name__ == "__main__":
    lims = (-2, 2)
    xs = ys = torch.linspace(lims[0], lims[1], 1000)
    Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")
    vecs = torch.stack((Xs, Ys), dim=-1)
    f = rosenbrock
    Zs = f(vecs)
    b_init = tensor([-2., -1])

    fig_3d, ax_3d = plt.subplots(
        subplot_kw=dict(projection="3d", computed_zorder=False)
    )
    ax_3d.plot_surface(Xs, Ys, Zs)
    ax_3d.set_xlim(lims[0], lims[1])
    ax_3d.set_ylim(lims[0], lims[1])

    # ax_3d.plot3D(1, 1, f(tensor([1.,1.])), "o", color="blue", markersize=5)

    (art_point,) = ax_3d.plot3D(
        b_init[0], b_init[1], f(b_init), "o", color="red", markersize=5
    )

    (art_line,) = ax_3d.plot3D([], [], [], color="red")

    Df = jacrev(f)
    Hf = hessian(f)

    a = torch.linspace(0, 10, 100)[...,None]
    b = b_init
    for i in range(50):
        print(f"=== {i} ===")
        print(
            f"Hessian if positive Definite : {(torch.linalg.eigvals(torch.squeeze(Hf(b))).real > 0).all()}"
        )

        Df_b = Df(b)
        print(f" norm of derivative: {torch.norm(Df_b)}")
        if (abs(norm(Df_b))) < 1e-3:
            break

        d = -torch.linalg.inv(torch.squeeze(Hf(b))) @ Df_b

        # draw line of search in 3d surface
        al = (b + a * d)
        Za = f(al)

        art_line.set_data_3d(*[torch.squeeze(x) for x in torch.split(al, 1, dim=1)], Za)
        wait()

        # b = line_search(f, Df, b, d)
        a = find_step_length(f, Df, b, d, plot=True)
        b = b+a*d
        print(f"new b : {b}")

        art_point.set_data_3d([b[0]], [b[1]], [f(b[None])])

    print(f"FINISHED, b_min = {b}")
