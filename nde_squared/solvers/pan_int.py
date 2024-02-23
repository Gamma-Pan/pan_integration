from typing import Iterable

import torch
from torch import tensor, Tensor
from ..optim import newton


def pan_int(f, y_init, t_lims: Iterable, step:torch.float, etol=1e-3) -> tuple[Tensor,Tensor]:
    def Phi_s(t: Tensor) -> Tensor:
        """
        This function returns a matrix Phi that when multiplied with the  coefficients matrix
        calculates the polynomial approximation of y(t) at the defined grid points

        :param t_lims: the initial and terminal time of integration
        :param y_init: the value of y at initial time
        :return: a tuple of the coefficients matrix and the Phi matrix
        """


    def Phi_c(t: Tensor) -> Tensor:
