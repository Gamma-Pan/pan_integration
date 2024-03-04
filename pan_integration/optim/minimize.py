from typing import Callable

import torch
from torch import sign, sqrt, abs, tensor, Tensor
from torch.func import jacfwd, jacrev
from pan_integration.utils.plotting import LsPlotter
from .factor import mod_chol


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


def _zoom(
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

        i = i + 1
        a_i = min_cubic_interpol(a_lo, phi_lo, Dphi_lo, a_hi, phi_hi, Dphi_hi)

        # evaluate phi at candidate min
        phi_i = phi(a_i)
        if phi_i > phi_0 + c1 * a_i * Dphi_0 or phi_i >= phi_lo:
            a_hi = a_i
        else:
            Dphi_i = Dphi(a_i)
            if abs(Dphi_i) <= c2 * Dphi_0:
                if plotting: plotter.close_all()
                return a_i

            if Dphi_i * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_i

    if plotting: plotter.close_all()
    return (a_lo + a_hi) / 2


def _line_search(
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

    def phi(a):
        return f(x + a * p)

    def Dphi(a):
        return p[None] @ grad_f(x + a * p)

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
        global plotter
        plotter = LsPlotter(phi)

    i = 1
    while i < max_iters and a_cur < a_max:
        # evaluate f at trial step (1 evaluation per loop)
        phi_cur = phi(a_cur)
        if plotting: plotter.line_search(a_cur)
        # if sufficient decrease is violated at new point find min with zoom
        if phi_cur > phi_0 + c1 * a_cur * Dphi_0 or (phi_cur >= phi_prev and i > 1):
            return _zoom(
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

        # if sufficient decrease holds and curvature hold return this a
        if abs(Dphi_cur) <= -c2 * Dphi_0:
            if plotting: plotter.close_all()
            return a_cur

        # TODO: explain this step
        if abs(Dphi_cur) >= 0:
            return _zoom(
                a_cur,
                a_prev,
                phi,
                Dphi,
                c1,
                c2,
                max_iters=5,
                phi_0=phi_0,
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

    if plotting: plotter.close_all()
    return a_cur


def newton(
        f: Callable,
        b_init: Tensor,
        f_args: tuple = None,
        f_kwargs: dict = None,
        has_aux=False,
        max_steps=50,
        metrics=True,
        tol: float = 1e-4,
        callback: Callable = None,
) -> Tensor:
    """
    :param f: scalar function to minimize
    :param b_init: tensor of parameters (tested only for 2d tensor)
    :param has_aux, whether the error function returns auxiliary outputs, error should be the first
    :param aux: the rest of the inputs to f
    :param max_steps: max amount of iterations to perform
    :param metrics: whether to print number of function evaluations
    :param tol: tolerance for termination
    :param callback: a function to call at end f each iteration
    :return: a b that is a local minimum of f
    """
    f_args = f_args or []
    f_kwargs = f_kwargs or {}
    num_coeff = torch.numel(b_init)

    # TODO: utilize jit
    def ff(x):
        res = f(x, *f_args, **f_kwargs)[0] if has_aux else f(x, *f_args, **f_kwargs)
        return res, res

    grad = jacrev(ff, has_aux=True)
    hessian = jacfwd(grad, has_aux=True)

    prev_norm = 0
    b = b_init
    for step in range(max_steps):

        if callback is not None:
            with torch.no_grad():
                callback(b.clone())

        # calculate forward pass and
        Df_k, f_k = grad(b)

        # check if gradient is within tolerance and if so return
        norm = torch.norm(Df_k)
        print(f"norm of grad: {norm} \t norms diff: {torch.abs(norm - prev_norm)}")
        if norm < tol or torch.abs(norm - prev_norm) < 1e-6:
            return b
        prev_norm = norm

        # calculate Hessian
        Hf_k = hessian(b)[0]

        # make hessian positive definite if not using modified Cholesky factorisation
        LD_compat = mod_chol(Hf_k, pivoting=False)
        D_ch = torch.diag(LD_compat)[:, None].clone()
        L_ch = torch.tril(LD_compat, -1).fill_diagonal_(1)

        # SOLVE LDL^T d_k = -Df_k
        #  first solve uni-triangular system 1
        Z = torch.linalg.solve_triangular(L_ch, -Df_k[:, None], upper=False, unitriangular=True)
        # then solve diagonal system 2
        Y = Z / D_ch  # + 1e-5) # for numerical stability
        # then sole uni-triangular system 1
        d_k = torch.linalg.solve_triangular(L_ch.T, Y, upper=True, unitriangular=True)

        # Df_k is 1d, make it a column vector to solve linear system
        # d_k = -torch.linalg.ldl_solve(LD_compat, torch.arange(1,num_coeff+1), Df_k[:, None])

        # if f_k < ff(b+0.00000001*d_k[:,0])[0] :
        #     raise Exception("NOT A DESCENT DIRECTION")

        alpha = _line_search(
            lambda x: ff(x)[0],
            lambda x: grad(x)[0],
            b,
            # local derivative is a batch of column vectors,
            d_k[:, 0],  # pass it to lineseach as a batch of 1d vectors
            phi_0=f_k,
            Dphi_0=(d_k.T @ Df_k).squeeze(),
            plot=False
        )

        b = b + alpha * d_k[:, 0]

    print("Max iterations reached")
    return b
