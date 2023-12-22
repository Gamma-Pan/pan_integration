import torch
from torch import tensor, abs, norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from nde_squared.optim.line_search import line_search
from torch.func import jacfwd, jacrev, hessian

mpl.use("TkAgg")


def wait():
    while True:
        if plt.waitforbuttonpress():
            break


def quadratic(vecs) -> tensor:
    A = tensor([[0.3, 3.4], [-10.7, 2.4]])
    b = tensor([[0.5, 0.5]])

    heights = vecs.transpose(-2, -1) @ A @ vecs + b @ vecs

    return torch.squeeze(heights)


def rosenborck(vecs) -> tensor:
    a = 1
    b = 100
    return (a - vecs[..., 0, 0]) ** 2 + b * (
        vecs[..., 1, 0] - vecs[..., 0, 0] ** 2
    ) ** 2


if __name__ == "__main__":
    lims = (-1, 1)
    xs = ys = torch.linspace(lims[0], lims[1], 200)
    Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")
    vecs = torch.stack((Xs, Ys), dim=2)[..., None]
    f = rosenborck
    Zs = f(vecs)
    b_init = tensor([[-.5], [-.5]])

    fig_3d, ax_3d = plt.subplots(
        subplot_kw=dict(projection="3d", computed_zorder=False)
    )
    ax_3d.plot_surface(Xs, Ys, Zs)
    ax_3d.set_xlim(lims[0], lims[1])
    ax_3d.set_ylim(lims[0], lims[1])

    (art_point,) = ax_3d.plot3D(
        b_init[0, 0], b_init[1, 0], f(b_init), "o", color="red", markersize=5
    )

    (art_line,) = ax_3d.plot3D([], [], [], color="red")

    Df = jacrev(f)
    Hf = hessian(f)

    a = torch.linspace(0, 10, 100)
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

        al = (b + a * d).transpose(1, 0)
        Za = f(al[..., None])

        art_line.set_data_3d(*[torch.squeeze(x) for x in torch.split(al, 1, dim=1)], Za)

        wait()

        print(f"current b : {b}")
        b = line_search(f, Df, b, d)
        print(f"new b : {b}")

        art_point.set_data_3d([b[0, 0]], [b[1, 0]], [f(b)])

    print(f"FINISHED, b_min = {b}")
