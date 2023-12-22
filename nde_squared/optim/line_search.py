import torch
from torch import sign, sqrt, abs
import matplotlib.pyplot as plt


def wait():
    while True:
        if plt.waitforbuttonpress():
            break


def backtracking(f, y, grad_y, direction, tau=0.5, alpha=1.0, c1=1e-3, max_iter=100):
    f_start = f(y)
    iter_cntr = 0
    while (
        f(y + alpha * direction) >= f_start + c1 * alpha * torch.dot(grad_y, direction)
        and iter_cntr < max_iter
    ):
        alpha = tau * alpha
        iter_cntr += 1
    return alpha


def min_cubic_interp(a0, phi0, D_phi0, a1, phi1, D_phi1):
    """
    Nocedal, Wright - Numerical Optimization 2006, pg. 59
    """
    d1 = D_phi0 + D_phi1 - 3 * (phi0 - phi1) / (a0 - a1)

    "check if sqrt of negative"
    if (d1 * d1 - D_phi0 * D_phi1) < 0.0:
        return 0.5 * (a1 + a0)

    d2 = sign(a1 - a0) * sqrt(d1 * d1 - D_phi0 * D_phi1)

    a_min = a1 - (a1 - a0) * (D_phi1 + d2 - d1) / (D_phi1 - D_phi0 + 2 * d2)

    if a1 > a0 and (a_min < a0 or a_min > a1):
        return 0.5 * (a1 + a0)
    elif a0 > a1 and (a_min < a1 or a_min > a0):
        return 0.5 * (a1 + a0)
    return torch.squeeze(a_min)


def zoom(
    a_lo,
    a_hi,
    phi,
    D_phi,
    c1=10e-4,
    c2=0.9,
    max_iters=5,
    phi_0=None,
    D_phi_0=None,
):
    if phi_0 is None:
        phi_0 = phi(0)
    if D_phi_0 is None:
        D_phi_0 = D_phi(0)

    if plotting:
        (li_lo,) = ax.plot([], [], "--", color="dodgerblue")
        (li_hi,) = ax.plot([], [], "--", color="dodgerblue")
        (art_min,) = ax.plot([], [], "o", color="red")

    i = 0
    while i < max_iters:
        print(a_lo, a_hi)

        if plotting:
            li_lo.set_data([a_lo, a_lo], [phi(a_lo) - 50.0, phi(a_lo) + 50.0])
            li_hi.set_data([a_hi, a_hi], [phi(a_hi) - 50.0, phi(a_hi) + 50.0])

        i = i + 1
        a_i = min_cubic_interp(
            a_lo, phi(a_lo), D_phi(a_lo), a_hi, phi(a_hi), D_phi(a_hi)
        )

        if plotting:
            art_min.set_data(a_i, phi(a_i))
            wait()

        phi_i = phi(a_i)
        if phi_i > phi_0 + c1 * a_i * D_phi_0 or phi_i >= phi(a_lo):
            a_hi = a_i
        else:
            D_phi_i = D_phi(a_i)
            if abs(D_phi_i) <= c2 * D_phi_0:
                return a_i

            if D_phi_i * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_i

    return (a_lo + a_hi) / 2


def strong_wolfe(
    phi=None,
    D_phi=None,
    a_i=torch.tensor(1),
    a_max=torch.tensor(10.0),
    max_iters=10,
    c1=1e-3,
    c2=0.9,
):
    i = 1
    a_prev = torch.tensor(0)
    phi_0 = phi(0)
    D_phi_0 = D_phi(0)
    while i < max_iters and a_i < a_max:
        phi_i = phi(a_i)

        if plotting:
            art_i.set_data(a_i, phi_i)
            wait()

        # sufficient decrease violated -> call zoom
        if phi_i > phi_0 + c1 * a_i * D_phi_0 or (phi_i >= phi(a_prev) and i > 1):
            return zoom(a_prev, a_i, phi, D_phi, c1, c2, phi_0=phi_0, D_phi_0=D_phi_0)

        D_phi_i = D_phi(a_i)

        if plotting:
            art_di.set_data(
                [a_i - 1.0, a_i + 1], [phi_i - 1 * D_phi_i, phi_i + 1 * D_phi_i]
            )
            wait()

        # sufficient decrease holds, curvature condition holds -> return a
        if abs(D_phi_i) <= c2 * abs(D_phi_0):
            return a_i

        # gradient position -> call zoom
        if D_phi_i >= 0:
            return zoom(a_i, a_prev, phi, D_phi, c1, c2, phi_0=phi_0, D_phi_0=D_phi_0)

        # larger interval
        a_prev = a_i
        a_i = torch.min(2 * a_i, a_max)
        i = i + 1

    return a_i


def line_search(f, Df, b, p, plot=False):
    global plotting
    plotting = plot
    def phi(a):
        return f(b + a * p)

    def D_phi(a):
        return torch.squeeze(Df(b + a * p).t() @ p)

    if plotting:
        global ax
        global fig
        fig, ax = plt.subplots()
        al = torch.linspace(0, 10, 1000)
        phi_al = phi(al[:, None, None])
        ax.plot(al, phi_al)
        ax.plot(0, phi(0), "go")
        # 1st wolfe condition
        ax.plot([0, 10], [phi(0), phi(0) + 10 * 1e-3 * D_phi(0)], "g--")
        # 2nd wolfe condition
        ax.plot([0, 2], [phi(0), phi(0) + 2 * 0.9 * D_phi(0)], "--", color="forestgreen")

        global art_i, art_di
        (art_i,) = ax.plot([], [], "o", color="orange")
        (art_di,) = ax.plot([], [], "--", color="darkorange")

        wait()

    a = strong_wolfe(phi, D_phi)
    print(f"step = {a}")

    if plotting:
        plt.close(fig)

    return b + a * p
