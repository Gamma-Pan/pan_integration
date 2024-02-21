import torch
from torch.func import jacfwd, jacrev
from nde_squared.optim.newton import minimize
from nde_squared.optim.factor import mod_chol


def f(B: torch.Tensor):
    return torch.sum(B[:, 0] ** 2 + B[:, 1] ** 2 + B[:, 2] ** 2) ** 2


def f_double(B: torch.Tensor):
    res = f(B)
    return res, res


if __name__ == "__main__":
    grad = jacrev(f_double, has_aux=True)
    hessian = jacfwd(lambda x: grad(x)[0])

    B_k = torch.ones((3, 3)).float()
    B_k[1, 2] = 10

    Df_k, f_k = grad(B_k)

    Hf_k = hessian(B_k)

    k = torch.arange(3)
    Hbatch = Hf_k[:, k, :, k]

    print(Hbatch)
    print(Df_k)
    x = torch.vstack([Hbatch[idx, :, :] @ Df_k[:, idx] for idx in [0, 1, 2]])

    print(' ')
    print(x)
    print(torch.tensordot(Hf_k, Df_k))
