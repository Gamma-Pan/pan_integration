import torch
import matplotlib.pyplot as plt
from math import sqrt
import sys


# def mod_chol(
#         A: Tensor, delta=tensor(1e-3), beta=tensor(10)) -> Tensor:
#     """
#     return the compact form of LDL^T Cholesky decomposition of the input matrix A
#     if A is positive definite, or a positive definite approximation if A is not
#     positive definite
#     """
#     n = A.shape[-1]  # row (= column) dimension
#     L = torch.zeros((n, n))
#     d = torch.zeros((n,))
#     C = torch.zeros(n, n)
#
#     for j in range(n - 1):
#         C[j, j] = A[j, j] - torch.sum(d[:j] * (torch.pow(L[j, :j], 2)))
#
#         theta = max(abs(C[j:, j]))
#         d[j] = max(torch.stack((abs(C[j, j]), delta, pow(theta / beta, 2))))
#
#         for i in range(j + 1, n):
#             C[i, j] = A[i, j] - torch.sum(d[:j] * L[i, :j] * L[j, :j])
#             L[i, j] = C[i, j] / d[j]
#
#     C_n = A[-1, -1] - torch.sum(d * (torch.pow(L[-1, :], 2)))
#     d[-1] = max(torch.stack((abs(C_n), delta)))
#
#     L[list(range(n)), list(range(n))] = d
#     return L


def mod_chol(G: torch.Tensor, delta=torch.tensor(1e-3)) -> tuple:
    """
    Implementing the Modified Cholesky Factorization as described in
    Gill, Murray, Wright - Practical Optimization pg.111
    """

    n = G.shape[0]
    nu = max(1., sqrt(n ** 2 - 1.))
    gamma = max(torch.diag(G))
    # G is symmetric so the max of the off diagonal elements is also the max
    # of the lower triangular part excluding the diagonal
    xi = torch.max(torch.tril(G,-1))
    eps = torch.tensor(sys.float_info.epsilon)
    beta_sq = torch.max(torch.stack((gamma, xi / nu, eps)))
    P = torch.arange(n)

    for j in range(n):
        # find the index of the largest element on the diagonal
        q = torch.argmax(torch.diag(G)[j:]).item()
        # interchange rows and columns of submatrix
        temp = P[j].clone()
        P[j] = q + j
        P[q + j] = temp

        temp = G[j:, j:][0, :].clone()
        G[j:, j:][0, :] = G[j:, j:][q, :]
        G[j:, j:][q, :] = temp

        temp = G[j:, j:][:, 0].clone()
        G[j:, j:][:, 0] = G[j:, j:][:, q]
        G[j:, j:][:, q] = temp

        # compute the j-th row of L
        G[j, :j] = G[j, :j] / torch.diagonal(G)[:j]
        # compute the j-th column of C and caclculate theta_j
        G[j + 1:, j] = G[j + 1:, j] - torch.sum(G[j, :j] * G[j + 1:, :j], dim=1)
        theta_j = torch.max(torch.abs(G[j + 1:, j])) if j < n - 1 else 0
        # compute the j-th diagonal element
        torch.diagonal(G)[j] = torch.max(torch.stack((delta, torch.abs(G[j, j]), theta_j ** 2 / beta_sq)))
        torch.diagonal(G)[j + 1:] -= G[j + 1:, j] / torch.diagonal(G)[j]

    return torch.tril(G).fill_diagonal_(1), torch.diagonal(G), P
