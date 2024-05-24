from typing import List
import torch
from torch import Tensor


def chain_mat_greedy(tensors: List[Tensor]) -> Tensor:
    if len(tensors) == 2:
        return tensors[0] @ tensors[1]
    elif len(tensors) == 1:
        return tensors[0]

    dims = [[*x.shape[-2:]] for x in tensors]
    # find the multiplication that requires the least calculations
    mults = torch.tensor([i[0] * i[1] * j[0] for i, j in zip(dims, dims[1:])])
    # mutliply the pair with the least calcs
    idx = torch.argmin(mults)

    new_tensors = tensors[:idx] + [tensors[idx] @ tensors[idx + 1]] + tensors[idx + 2 :]
    del tensors[idx]
    del tensors[idx]

    return chain_mat_greedy(new_tensors)
