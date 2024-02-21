import torch
from torch import tensor, Tensor
from torch.func import jacrev, jacfwd
from typing import Callable

from .factor import mod_chol
from .line_search import find_step_length


def minimize(
    f: Callable,
    b_init: Tensor,
    max_steps=50,
    metrics=True,
    tol: float = 1e-4,
    callback: Callable = None,
):
    """
    :param f: scalar function to minimize
    :param b_init: tensor of parameters (tested only for 2d tensor)
    :param max_steps: max amount of iterations to perform
    :param metrics: whether to print number of function evaluations
    :param tol: tolerance for termination
    :param callback: a function to call at end of each iteration
    :return: a b that is a local minimum of f
    """
    num_coeff = torch.numel(b_init)

    # TODO: utilize jit
    def ff(x):
        res = f(x.squeeze()[None]).squeeze()
        return res, res

    grad = jacrev(ff, has_aux=True)
    hessian = jacfwd(lambda x: grad(x)[0])

    b = b_init.squeeze()
    for step in range(max_steps):
        # calculate forward pass and
        Df_k, f_k = grad(b)

        # check if gradient is within tolerance and if so return
        if torch.norm(Df_k) < tol:
            print()
            return b

        # calculate Hessian
        Hf_k = hessian(b)
        # make hessian positive definite if not using modified Cholesky factorisation
        LD_compat = mod_chol(Hf_k)

        # pivots for solving LDL problem,
        # since factorisation doesn't use permutations just use a range
        pivots = torch.arange(1, num_coeff + 1)
        d_k = -torch.linalg.ldl_solve(LD_compat, pivots, Df_k[:,None])

        alpha = find_step_length(
            lambda x: ff(x)[0], lambda x: grad(x)[0], b[:,None], d_k, phi_0=f_k, Dphi_0=d_k.T @Df_k
        )

        b = b + alpha * d_k.squeeze()

        if callback is not None:
            callback(b, d_k)

    print("Max iterations reached")
    return b
