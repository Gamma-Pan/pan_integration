import torch
import matplotlib.pyplot as plt
from math import sqrt
import sys
from typing import Union


def mod_chol(G: torch.Tensor, delta=torch.tensor(1e-3), pivoting=False) -> Union[tuple, torch.Tensor]:
    """
    Implementing the Modified Cholesky Factorization as described in
    Gill, Murray, Wright - Practical Optimization pg.111
    """
    # TODO: use the same storage for L and d

    n = G.shape[0]

    L = torch.zeros(n, n)
    d = torch.diag(G)

    nu = max(1., sqrt(n ** 2 - 1.))
    gamma = torch.max(torch.diag(G))
    # G is symmetric so the max of the off diagonal elements is also the max
    # of the lower triangular part excluding the diagonal
    xi = torch.max(torch.tril(G, -1))
    eps =  torch.tensor(1e-10)  # torch.tensor(sys.float_info.epsilon)
    beta_sq = torch.max(torch.stack((gamma, xi / nu, eps)))
    P = torch.arange(n)

    for j in range(n):

        if pivoting:
            # find the index of the largest element on the diagonal
            q = torch.argmax(d[j:]).item()
            # interchange rows and columns of G

            # swap pivots
            temp = P[j].clone()
            P[j] = P[q + j]
            P[q + j] = temp

            # swap rows of G
            temp = G[j, :].clone()
            G[j, :] = G[q + j, :]
            G[q + j, :] = temp

            # swap columns of G
            temp = G[:, j].clone()
            G[:, j] = G[:, q + j]
            G[:, q + j] = temp

            # swap rows of L
            temp = L[j, :].clone()
            L[j, :] = L[q + j, :]
            L[q + j, :] = temp

            # swap elements of d
            temp = d[j].clone()
            d[j] = d[q + j]
            d[q + j] = temp

        # compute the j-th row of L
        L[j, :j] = L[j, :j] / d[:j]
        # compute the j-th column of C and calculate theta_j
        L[j + 1:, j] = G[j + 1:, j] - torch.sum(L[j, :j] * L[j + 1:, :j], dim=1)
        theta_j = torch.max(torch.abs(L[j + 1:, j])) if j < n - 1 else 0
        # compute the j-th diagonal element
        d[j] = torch.max(torch.stack((delta, torch.abs(d[j]), theta_j ** 2 / beta_sq)))

        d[j + 1:] -= L[j + 1:, j] ** 2 / d[j]

    torch.diagonal(L)[:] = d
    if pivoting:
        return L, P
    else:
        return L
