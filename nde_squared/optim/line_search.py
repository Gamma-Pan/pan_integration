import torch
from torch import sign, sqrt, abs


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
    max_iters=10,
    phi_0=None,
    D_phi_0=None,
    ax=None,
):
    if phi_0 is None:
        phi_0 = phi(0)
    if D_phi_0 is None:
        D_phi_0 = D_phi(0)

    i = 0
    while i < max_iters:
        i = i + 1
        a_i = min_cubic_interp(
            a_lo, phi(a_lo), D_phi(a_lo), a_hi, phi(a_hi), D_phi(a_hi)
        )
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
    a=torch.tensor(1),
    a_max=torch.tensor(2.),
    max_iters=10,
    c1=1e-3,
    c2=0.9,
    ax=None,
):
    i = 1
    a_prev = torch.tensor(0)
    phi_0 = phi(0)
    D_phi_0 = D_phi(0)
    while i < max_iters:
        phi_i = phi(a)
        if phi_i > phi_0 + c1 * a * D_phi_0 or (phi_i >= phi(a_prev) and i > 1):
            return zoom(a_prev, a, phi, D_phi, c1, c2, phi_0=phi_0, D_phi_0=D_phi_0)
        D_phi_i = D_phi(a)
        if abs(D_phi_i) <= -c2 * D_phi_0:
            return a
        if D_phi_i >= 0:
            return zoom(a, a_prev, phi, D_phi, c1, c2, phi_0=phi_0, D_phi_0=D_phi_0)
        a_prev = a
        a = torch.min(2.0 * a, a_max)
        i = i + 1

    return a


def line_search(f, Df, b, p, ax=None):
    def phi(a):
        return f(b + a * p)

    def D_phi(a):
        return Df(b + a * p).t() @ p

    a = strong_wolfe(phi, D_phi, ax=ax)

    return b + a * p
