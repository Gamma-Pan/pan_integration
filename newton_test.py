import torch
from torch.func import jacfwd, jacrev
from nde_squared.optim.newton import minimize
from nde_squared.optim.factor import mod_chol


def f(B: torch.Tensor):
    return B[0, 1]**2 * B[1, 1]**2


def f_double(B: torch.Tensor):
    res = f(B)
    return res, res


if __name__ == "__main__":
    grad = jacrev(f_double, has_aux=True)
    hessian = jacfwd(lambda x: grad(x)[0])

    B_k = torch.ones((2, 2)).float()

    Df_k, f_k = grad(B_k)

    Hf_k = hessian(B_k)

    print(B_k)
    print(" ")
    print(Hf_k)
    print(" ")
    print(Hf_k[:, 1, :, 1])
