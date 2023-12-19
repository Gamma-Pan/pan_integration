import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from nde_squared.optim.line_search import line_search
from torch.func import jacfwd, jacrev

mpl.use("TkAgg")


def quadratic(vecs) -> torch.Tensor:
    A = torch.tensor([[7.0, 5.0], [-7.0, 5.0]])
    b = torch.tensor([[0.2, 0.3]])

    heights = vecs.transpose(-2, -1) @ A @ vecs + b @ vecs

    return torch.squeeze(heights)


if __name__ == "__main__":
    xs = ys = torch.linspace(-5, 5, 200)
    Xs, Ys = torch.meshgrid(xs, ys, indexing="xy")
    vecs = torch.stack((Xs, Ys), dim=2)[..., None]
    Zs = quadratic(vecs)
    b_init = torch.tensor([[4.0], [4.0]])

    fig_3d, ax_3d = plt.subplots(
        subplot_kw=dict(projection="3d", computed_zorder=False)
    )
    ax_3d.plot_surface(Xs, Ys, Zs)
    ax_3d.scatter3D(
        b_init[0, 0],
        b_init[1, 0],
        quadratic(b_init),
        color="red",
        s=20,
    )

    Df = jacrev(quadratic)
    Hf = jacfwd(jacrev(quadratic))

    d = -torch.linalg.inv(torch.squeeze(Hf(b_init))) @ Df(b_init)

    b_next = line_search(quadratic, Df, b_init, d)

    print(b_next)

    def phi(a):
        return quadratic(b_init + a * d)

    a = torch.linspace(0, 2, 100)
    al = (b_init + a * d).transpose(1, 0)
    Za = quadratic(al[..., None])

    ax_3d.plot3D(*[torch.squeeze(x) for x in torch.split(al, 1, dim=1)], Za)
    fig_1d, ax_1d = plt.subplots()
    ax_1d.plot(a, Za)
    ax_1d.scatter([0], [Za[0]], marker="o", color='red')

    plt.show()
