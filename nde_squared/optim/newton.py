import torch
from torch import tensor, Tensor
from torch.func import jacrev, jacfwd
from typing import Callable

from .factor import mod_chol
from .line_search import find_step_length

def minimize(f : Callable, b_init: Tensor, max_steps=50, metrics = True, tol:float= 1e-4):

    # TODO: utilize jit
    grad = jacrev(lambda x: (f(x), f(x)), has_aux=True)

    # returns the required derivatives and the forward pass
    # in the form of   (Hf, (Df, f))
    derivatives = jacfwd(lambda x: (grad(x), grad(x)), has_aux=True)

    b_k = b_init
    for step in range(max_steps):
        (H_k, (D_k, f_k)) = derivatives(b_k)

        # check if gradient is within tolerance
        if torch.norm(D_k) < tol:
            return b_k

        # if hessian is not positive definite modify it
        H_k = mod_chol(H_k)

        # calculate gradient direction
        pivots = torch.arange(1, )





