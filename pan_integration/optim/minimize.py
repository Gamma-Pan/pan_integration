from typing import Callable, Any, Tuple, Iterable, Dict

import torch
from torch import sign, sqrt, abs, tensor, Tensor, stack, dot, cat
from torch.func import grad, hessian, vmap
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
    return a_min


def _zoom(
    a_lo,
    a_hi,
    phi,
    Dphi,
    c1=10e-4,
    c2=0.9,
    max_iters=20,
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
        if plotting:
            plotter.zoom(a_i, c1, c2, Dphi_0)
        phi_i = phi(a_i)
        Dphi_i = Dphi(a_i)
        if phi_i > phi_0 + c1 * a_i * Dphi_0 or phi_i >= phi_lo:
            a_hi = a_i
            phi_hi = phi_i
            Dphi_hi = Dphi_hi
        else:
            if abs(Dphi_i) <= -c2 * Dphi_0:
                if plotting:
                    plotter.close_all()
                return a_i

            if Dphi_i * (a_hi - a_lo) >= 0:
                a_hi = a_lo
                phi_hi = phi_lo
                Dphi_hi = Dphi_lo
            a_lo = a_i
            phi_lo = phi_i
            Dphi_lo = Dphi_i

    if plotting:
        plotter.close_all()
    return (a_lo + a_hi) / 2


def _line_search(
    f: Callable,
    grad_f: Callable,
    x: Tensor,
    p: Tensor,
    c1=1e-3,
    c2=0.9,
    max_iters=20,
    max_zoom_iters=5,
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
        return dot(p, grad_f(x + a * p))

    if phi_0 is None or Dphi_0 is None:
        phi_0 = phi(0)
        Dphi_0 = Dphi(0)

    a_prev = torch.tensor(0.0)
    a_cur = torch.tensor(1)
    phi_prev = phi_0
    Dphi_prev = Dphi_0

    global plotting
    plotting = plot
    if plotting:
        global plotter
        plotter = LsPlotter(phi, Dphi)

    i = 1
    while i < max_iters:
        # evaluate f at trial step (1 evaluation per loop)
        phi_cur = phi(a_cur)
        if plotting:
            plotter.line_search(a_cur, c1)
        # if sufficient decrease is violated at new point find min with zoom
        if phi_cur > phi_0 + c1 * a_cur * Dphi_0 or phi_cur >= phi_prev and i > 1:
            return _zoom(
                a_prev,
                a_cur,
                phi,
                Dphi,
                c1,
                c2,
                phi_0=phi_0,
                Dphi_0=Dphi_0,
                phi_lo=phi_prev,
                Dphi_lo=Dphi_prev,
                phi_hi=phi_cur,
                max_iters=max_zoom_iters
            )

        # evaluate grad f at trial step (~2 evaluations per loop)
        Dphi_cur = Dphi(a_cur)

        # if sufficient decrease holds and curvature hold return this a
        if abs(Dphi_cur) <= -c2 * Dphi_0:
            if plotting:
                plotter.close_all()
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
                phi_0=phi_0,
                Dphi_0=Dphi_0,
                phi_lo=phi_cur,
                Dphi_lo=Dphi_cur,
                phi_hi=phi_prev,
                Dphi_hi=Dphi_prev,
                max_iters=max_zoom_iters
            )

        # increase step size
        phi_prev = phi_cur
        Dphi_prev = Dphi_cur
        a_prev = a_cur
        a_cur = 2 * a_cur
        i = i + 1

    if plotting:
        plotter.close_all()
    return a_cur


def newton(
    f: Callable,
    b_init: Tensor,
    f_args: Iterable=None,
    f_kwargs: Dict=None,
    max_steps=50,
    max_linesearch_iters=10,
    max_zoom_iters=10,
    etol: float = 1e-7,
    plot= False,
    callback: Callable = None,
) -> Tensor | tuple[Tensor | Any]:
    """
    :param f: scalar function to minimize
    :param b_init: tensor of parameters (tested only for 2d tensor)
    :param has_aux, whether the error function returns auxiliary outputs, error should be the first
    :param max_steps: max amount of iterations to perform
    :param etol: tolerance for termination
    :param callback: a function to call at end of each iteration
    :return: b* that is a local minimum of f
    """
    if f_args is None:
        f_args =()
    if f_kwargs is None:
        f_kwargs = {}

    batches, dims = b_init.shape

    jac = vmap(grad(f))
    hess = vmap(hessian(f))

    b = b_init
    prev_grad_norm = 0

    for step in range(max_steps):
        if callback is not None:
            with torch.no_grad():
                callback(b.cpu())

        Df_all = jac(b, *f_args, **f_kwargs)

        grad_norm = torch.norm(Df_all)
        # check if gradient is within tolerance and if so return
        if grad_norm < etol or abs(grad_norm - prev_grad_norm) < etol:
            return b

        prev_grad_norm = grad_norm

        # calculate Hessian
        Hf_all = hess(b, *f_args, **f_kwargs)

        # p_all = []
        # for Hf_k, Df_k in zip(Hf_all, Df_all):
        #     # make hessian positive definite if not using modified Cholesky factorisation
        #     LD_compat = mod_chol(Hf_k, pivoting=False)
        #     D_ch = torch.diag(LD_compat)[:, None].clone()
        #     L_ch = torch.tril(LD_compat, -1).fill_diagonal_(1)
        #
        #     # SOLVE LDL^T d_k = -Df_k
        #     #  first solve uni-triangular system
        #     Z = torch.linalg.solve_triangular(
        #         L_ch, -Df_k[:, None], upper=False, unitriangular=True
        #     )
        #     # then solve diagonal system
        #     Y = Z / D_ch  # + 1e-5) # for numerical stability
        #     # then solve uni-triangular system
        #     p_all.append(
        #         torch.linalg.solve_triangular(
        #             L_ch.T, Y, upper=True, unitriangular=True
        #         )[:, 0]
        #     )
        #
        # p_k = torch.cat(p_all, dim=0)

        p_k = torch.linalg.solve( Hf_all,-Df_all ).reshape(-1)

        def batch_f(b_vec):
            b = b_vec.reshape(batches, dims)
            return torch.sum(  vmap(f)(b, *f_args, **f_kwargs)  )

        def batch_jac(b_vec):
            b = b_vec.reshape(batches, dims)
            return jac(b, *f_args, **f_kwargs).reshape(-1)

        b_vec = b.reshape(-1)
        alpha = _line_search(
            batch_f,
            batch_jac,
            b_vec,
            p_k,
            phi_0=batch_f(b_vec),
            Dphi_0=dot(p_k, Df_all.reshape(-1)),
            plot=plot,
            max_iters=max_linesearch_iters,
            max_zoom_iters=max_zoom_iters
        )
        b = b + alpha * p_k.reshape(batches, dims)
        print(alpha)

    print("Max iterations reached")
    return b.reshape(batches, dims)
