import torch
from torch import sign, sqrt, abs, tensor, Tensor, pow, max
import matplotlib.pyplot as plt


def mod_chol(
    A: Tensor, delta=tensor(1e-3), beta_sq=tensor(10)) -> Tensor :
    """
    return the compact form of LDL^T Cholesky decomposition of the input matrix A
    if A is positive definite, or a positive definite approximation if A is not
    positive definite
    """
    n = A.shape[-1]  # row (= column) dimension
    L = torch.zeros((n, n))
    d = torch.zeros((n,))
    C = torch.zeros(n, n)

    for j in range(n - 1):
        C[j, j] = A[j, j] - torch.sum(d[:j] * (torch.pow(L[j, :j], 2)))
        theta = max(abs(C[j + 1 :, j]))
        d[j] = max(torch.stack((abs(C[j, j]), delta, pow(theta, 2) / beta_sq)))

        for i in range(j + 1, n):
            C[i, j] = A[i, j] - torch.sum(d[:j] * L[i, :j] * L[j, :j])
            L[i, j] = C[i, j] / d[j]

    C_n = A[-1, -1] - torch.sum(d * (torch.pow(L[-1, :], 2)))
    d[-1] = max(torch.stack((abs(C_n), delta)))

    L[list(range(n)) , list(range(n)) ] = d
    return L


def mod_chol_permute(A: Tensor, delta=tensor(1e-3), beta=tensor(10e3)):
    raise NotImplementedError
