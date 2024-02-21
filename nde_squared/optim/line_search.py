from typing import Callable

import torch
from torch import sign, sqrt, abs, tensor, Tensor
import matplotlib.pyplot as plt
from nde_squared.utils.plotting import wait


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


def min_cubic_interpol(a0, phi0, D_phi0, a1, phi1, D_phi1):
    """
    Nocedal, Wright - Numerical Optimization 2006, pg. 59
    """
    if a0 == a1:
        return a0

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
    Dphi,
    c1=10e-4,
    c2=0.9,
    max_iters=5,
    phi_0=None,
    Dphi_0=None,
    phi_lo=None,
    Dphi_lo=None,
    phi_hi=None,
    Dphi_hi=None,
):
    if phi_0 is None:
        phi_0 = phi(0)
    if Dphi_0 is None:
        Dphi_0 = Dphi(0)
    if phi_lo is None:
        phi_lo = phi(a_lo)
    if phi_hi is None:
        phi_hi = phi(a_hi)
    if Dphi_hi is None:
        Dphi_hi = Dphi(a_hi)
    if Dphi_lo is None:
        Dphi_lo = Dphi(a_lo)

    i = 0
    while i < max_iters:
        if plotting:
            ax.plot(a_lo, phi_lo, color="green", marker="o")
            ax.plot(a_hi, phi_hi, color="blue", marker="o")
            wait()

        i = i + 1
        a_i = min_cubic_interpol(a_lo, phi_lo, Dphi_lo, a_hi, phi_hi, Dphi_hi)

        # evaluate phi at candidate min
        phi_i = phi(a_i)
        if phi_i > phi_0 + c1 * a_i * Dphi_0 or phi_i >= phi_lo:
            a_hi = a_i
        else:
            Dphi_i = Dphi(a_i)
            if abs(Dphi_i) <= c2 * Dphi_0:
                return a_i

            if Dphi_i * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_i

    return (a_lo + a_hi) / 2


def find_step_length(
    f: Callable,
    grad_f: Callable,
    x: Tensor,
    p: Tensor,
    c1=1e-3,
    c2=0.9,
    a_max=tensor(10.0),
    max_iters=10,
    phi_0=None,
    Dphi_0=None,
    plot=False,
):
    """
    Algorithm 3.6 from Nocedal, Wright - Numerical Approximation pg.62; returns
    an approximation of the step length a, along direction p, that minimises f(x + a*p)
    using strong Wolfe conditions

    """

    phi = lambda a: f(x + a * p)
    Dphi = lambda a: p.T @ grad_f(x + a * p)

    if phi_0 is None or Dphi_0 is None:
        phi_0 = phi(0)
        Dphi_0 = Dphi(0)

    a_prev = tensor(0)
    a_cur = tensor(1)
    phi_prev = phi_0
    Dphi_prev = Dphi_0

    global plotting
    plotting = plot
    if plotting:
        global ax
        fig, ax = plt.subplots()

        a_plot = torch.linspace(0, 5, 100)[:, None]
        ax.plot(a_plot.squeeze(), phi(a_plot).squeeze(), color="red", label="\phi (a)")
        (art_slope0,) = ax.plot(a_plot, torch.zeros(*a_plot.shape), linestyle="--")
        (art_point,) = ax.plot([], [], color="green", marker="o")
        (art_slopei,) = ax.plot([], [], linestyle="-.", color="red")
        (art_slopeiabs,) = ax.plot([], [], linestyle="-.", color="seagreen")
        (art_slopei0,) = ax.plot([], [], linestyle="-.", color="lime")

    i = 1
    while i < max_iters and a_cur < a_max:
        # evaluate f at trial step (1 evaluation per loop)
        phi_cur = phi(a_cur)

        if plotting:
            art_slope0.set_ydata(phi_0 + c1 * a_plot * Dphi_0)
            art_point.set_data(a_cur, phi_cur)
            wait()

        # if sufficient decrease is violated at new point find min with zoom
        if phi_cur > phi_0 + c1 * a_cur * Dphi_0 or (phi_cur >= phi_prev and i > 1):
            if plotting:
                art_slope0.set_data([], [])
                art_point.set_data([], [])

            return zoom(
                a_prev,
                a_cur,
                phi,
                Dphi,
                c1,
                c2,
                max_iters=5,
                phi_0=phi_0,
                Dphi_0=Dphi_0,
                phi_lo=phi_prev,
                Dphi_lo=Dphi_prev,
            )

        # evaluate grad f at trial step (~2 evaluations per loop)
        Dphi_cur = Dphi(a_cur)

        if plotting:
            art_slopeiabs.set_data(
                [a_cur - 1, a_cur + 1],
                [phi_cur - torch.abs(Dphi_cur), phi_cur + torch.abs(Dphi_cur)],
            )
            art_slopei.set_data(
                [a_cur - 1, a_cur + 1],
                [phi_cur - Dphi_cur, phi_cur + Dphi_cur],
            )
            art_slopei0.set_data(
                [a_cur - 1, a_cur + 1], [phi_cur + c2 * Dphi_0, phi_cur - c2 * Dphi_0]
            )
            wait()

        # if sufficient decrease holds and curvature hold return this a
        if abs(Dphi_cur) <= -c2 * Dphi_0:
            if plotting:
                plt.close(fig)
            return a_cur

        if plotting:
            art_slopei.set_data([], [])
            art_slopeiabs.set_data([], [])
            art_slopei0.set_data([], [])

        # TODO: explain this step
        if abs(Dphi_cur) >= 0:
            return zoom(
                a_cur,
                a_prev,
                phi,
                Dphi,
                c1,
                c2,
                max_iters=5,
                phi_0=(phi_0,),
                Dphi_0=Dphi_0,
                phi_lo=phi_cur,
                Dphi_lo=Dphi_cur,
                phi_hi=phi_prev,
                Dphi_hi=Dphi_prev,
            )

        # increase step size
        phi_prev = phi_cur
        Dphi_prev = Dphi_cur
        a_prev = a_cur
        a_cur = 2 * a_cur
        i = i + 1

    return a_cur
