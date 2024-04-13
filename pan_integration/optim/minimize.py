from typing import Callable, Any, Tuple

import torch
from torch import sign, sqrt, abs, tensor, Tensor, stack
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
    return torch.squeeze(a_min)


def _zoom_vec(
    a_lo,
    a_hi,
    phi,
    Dphi,
    c1=10e-4,
    c2=0.9,
    max_iters=10,
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

    batches = a_lo.shape[0]

    a_i_out = torch.zeros(batches)
    mask_done = torch.zeros(batches)

    i = 0
    while i < max_iters:
        i = i + 1

        # no f evals in here, loop freely

        if a_lo.size(0) == 1:
            a_i = torch.stack(
                [
                    min_cubic_interpol(*args)
                    for args in zip(
                        a_lo[None],
                        phi_lo[None],
                        Dphi_lo[None],
                        a_hi[None],
                        phi_hi[None],
                        Dphi_hi[None],
                    )
                ]
            )
        else:
            a_i = torch.stack(
                [
                    min_cubic_interpol(*args)
                    for args in zip(a_lo, phi_lo, Dphi_lo, a_hi, phi_hi, Dphi_hi)
                ]
            )

        phi_i = phi(a_i)

        mask_cond1 = (phi_i > phi_0 + c1 * a_i * Dphi_0).logical_or(phi_i >= phi_lo)
        a_hi[mask_cond1] = a_i[mask_cond1]

        Dphi_i = Dphi(a_i)
        mask_cond2 = abs(Dphi_i) <= -c2 * Dphi_0
        a_i_out[mask_cond1.logical_not().logical_and(mask_cond2)] = a_i[
            mask_cond1.logical_not().logical_and(mask_cond2)
        ]
        mask_done[mask_cond1.logical_not().logical_and(mask_cond2)] = 1

        if torch.all(mask_done):
            return a_i_out

        mask_cond3 = Dphi_i * (a_hi - a_lo) >= 0
        a_hi[mask_cond1.logical_not().logical_and(mask_cond3)] = a_lo[
            mask_cond1.logical_not().logical_and(mask_cond3)
        ]

        a_lo[mask_cond1.logical_not()] = a_i[mask_cond1.logical_not()]

    a_i_out[mask_done.logical_not()] = (
        a_lo[mask_done.logical_not()] + a_hi[mask_done.logical_not()]
    ) / 2
    return a_i_out


def _zoom(
    a_lo,
    a_hi,
    phi,
    Dphi,
    c1=10e-4,
    c2=0.9,
    max_iters=10,
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
        if phi_i > phi_0 + c1 * a_i * Dphi_0 or phi_i >= phi_lo:
            a_hi = a_i
        else:
            Dphi_i = Dphi(a_i)
            if abs(Dphi_i) <= -c2 * Dphi_0:
                if plotting:
                    plotter.close_all()
                return a_i

            if Dphi_i * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_i

    if plotting:
        plotter.close_all()
    return (a_lo + a_hi) / 2


def _line_search_vec(
    f: Callable,
    grad_f: Callable,
    x: Tensor,
    p: Tensor,
    c1=1e-3,
    c2=0.9,
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
    batches, dim = x.shape

    def phi(a,x,p):
        return f(x + (a[:, None, None] * p)[...,0])

    def Dphi(a,x,p):
        return (p.mT @ grad_f(x + (a[:, None, None] * p)[...,0])[...,None])[:,0,0]

    if phi_0 is None or Dphi_0 is None:
        phi_0 = phi(0)
        Dphi_0 = Dphi(0)

    mask_done = torch.zeros(batches).to(torch.bool)

    a_prev = torch.zeros(batches)
    a_cur = torch.ones(batches)
    phi_prev = phi_0
    Dphi_prev = Dphi_0

    i = 1
    while i < max_iters:
        phi_cur = phi(a_cur)
        Dphi_cur = Dphi(a_cur)

        mask_violates_w1 = mask_done.logical_not().logical_and(
            (phi_cur > phi_0 + c1 * a_cur * Dphi_0).logical_or(
                (phi_cur >= phi_prev).logical_and(tensor(torch.ones(batches) * i > 1))
            )
        )
        # these are done
        mask_holds_w2 = mask_done.logical_not().logical_and(
            mask_violates_w1.logical_not().logical_and(abs(Dphi_cur) <= -c2 * Dphi_0)
        )
        mask_done[mask_holds_w2[0]] = True
        if torch.all(mask_done):
            return a_cur

        mask_violates_w2 = mask_done.logical_not().logical_and(
            mask_holds_w2.logical_not()
            .logical_and(mask_violates_w1.logical_not())
            .logical_and(abs(Dphi_cur) >= 0)
        )

        mask_zoom = mask_violates_w1.logical_or(mask_violates_w2)[0]
        zoom_range = torch.arange(torch.sum(mask_zoom).to(torch.int))
        a_cur[mask_zoom] = _zoom_vec(
            torch.stack((a_cur[mask_zoom], a_prev[mask_zoom]))[
                mask_violates_w1[mask_zoom].to(torch.long), zoom_range
            ],
            torch.stack((a_cur[mask_zoom], a_prev[mask_zoom]))[
                mask_violates_w1[mask_zoom].logical_not().to(torch.long), zoom_range
            ],
            lambda a: f(x[mask_zoom, :] + a[:, None] * p[mask_zoom, :]),
            lambda a: torch.squeeze(
                p[mask_zoom, :][:, None, :]
                @ grad_f(x[mask_zoom, :] + a[:, None] * p[mask_zoom, :])[:, :, None]
            ),
            c1,
            c2,
            max_iters=5,
            phi_0=phi_0[mask_zoom],
            Dphi_0=Dphi_0[mask_zoom],
            phi_lo=phi_prev[mask_zoom],
            Dphi_lo=Dphi_prev[mask_zoom],
        )
        mask_done[mask_zoom] = True

        # increase step size
        phi_prev = phi_cur
        Dphi_prev = Dphi_cur
        a_prev[mask_done.logical_not()] = a_cur[mask_done.logical_not()]
        a_cur[mask_done.logical_not()] = 2 * a_cur[mask_done.logical_not()]
        i = i + 1

    return a_cur


def _line_search(
    f: Callable,
    grad_f: Callable,
    x: Tensor,
    p: Tensor,
    c1=1e-3,
    c2=0.9,
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
    batches, dim = x.shape

    def phi(a):
        return f(x + a * p)

    def Dphi(a):
        return torch.squeeze(p @ grad_f(x + a * p).mT)

    if phi_0 is None or Dphi_0 is None:
        phi_0 = phi(0)
        Dphi_0 = Dphi(0)

    # extremely convoluted way to vectorize line search to multiple batches
    mask = torch.ones(batches)

    a_prev = torch.tensor(0.0)
    a_cur = torch.tensor(1.0)
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

    if plotting:
        plotter.close_all()
    return a_cur


def newton(
    f: Callable,
    b_init: Tensor,
    has_aux=False,
    modify=True,
    max_steps=50,
    etol: float = 1e-5,
    callback: Callable = None,
) -> Tensor | tuple[Tensor | Any]:
    """
    :param f: scalar function to minimize
    :param b_init: tensor of parameters (tested only for 2d tensor)
    :param has_aux, whether the error function returns auxiliary outputs, error should be the first
    :param modify: modify Hessian to be positive definite, if False assume it is
    :param max_steps: max amount of iterations to perform
    :param etol: tolerance for termination
    :param callback: a function to call at end of each iteration
    :return: b* that is a local minimum of f
    """

    # define this function to get both forward pass and gradients from a single jacrev call
    # https://pytorch.org/functorch/stable/generated/functorch.jacrev.html#:~:text=If%20you%20would%20like%20to%20compute%20the%20output%20of%20the%20function%20as%20well%20as%20the%20jacobian%20of%20the%20function%2C%20use%20the%20has_aux%20flag%20to%20return%20the%20output%20as%20an%20auxiliary%20object%3A
    # TODO: utilize jit
    def ff(x):
        if has_aux:
            res, aux = f(x)
            return res, (res, aux)
        else:
            res = f(x)
            return res, res

    jac = vmap(grad(ff, has_aux=True))
    hess = vmap(hessian(ff))

    b = b_init
    prev_grad_norm = 0

    for step in range(max_steps):
        if callback is not None:
            with torch.no_grad():
                callback(b.cpu())

        # calculate forward pass and jacobian in one pass
        Df_k, aux = jac(b)
        if has_aux:
            f_k = aux[0]
        else:
            f_k = aux

        grad_norm = torch.norm(Df_k)
        # check if gradient is within tolerance and if so return
        if grad_norm < etol or abs(grad_norm - prev_grad_norm) < etol:
            return (
                (
                    b,
                    aux[1:],
                )
                if has_aux
                else b
            )

        prev_grad_norm = grad_norm

        # calculate Hessian
        Hf_k = hess(b)[0]

        # WARNING: POSSIBLE BOTTLENECK, it's a pain to vectorize this so I loop in batch
        # if I start from a convex region this can be skipped

        if modify:
            d_k = []
            for H, D in zip(Hf_k.unbind(dim=0), Df_k.unbind(dim=0)):
                # make hessian positive definite if not using modified Cholesky factorisation
                LD_compat = mod_chol(H, pivoting=False)
                D_ch = torch.diag(LD_compat)[:, None].clone()
                L_ch = torch.tril(LD_compat, -1).fill_diagonal_(1)

                # SOLVE LDL^T d_k = -Df_k
                #  first solve uni-triangular system
                Z = torch.linalg.solve_triangular(
                    L_ch, -D[:, None], upper=False, unitriangular=True
                )
                # then solve diagonal system
                Y = Z / D_ch  # + 1e-5) # for numerical stability
                # then solve uni-triangular system
                d_k.append(
                    torch.linalg.solve_triangular(
                        L_ch.T, Y, upper=True, unitriangular=True
                    )
                )

            d_k = torch.stack(d_k)
        else:
            d_k = torch.linalg.solve(Hf_k, Df_k)

        alpha = _line_search_vec(
            lambda x: ff(x)[0],
            lambda x: jac(x)[0],
            b,
            # local derivative is a batch of column vectors,
            d_k,
            phi_0=f_k,
            Dphi_0=(d_k.mT @ Df_k[..., None])[:,0,0],
            plot=True,
        )

        b = b + alpha[:, None] * d_k[..., 0]
        # b = b + alpha * d_k[..., 0]

    print("Max iterations reached")
    if has_aux:
        return b, aux[1:]
    else:
        return b
