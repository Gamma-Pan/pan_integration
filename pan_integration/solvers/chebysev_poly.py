import torch


def T_grid(num_t, num_coeff, t0, t1):
    out = torch.empty(num_t, num_coeff)
    out[:,0 ] = torch.ones( num_t,1 )
    out[:,1] = torch.linspace( -1. ,1, )

    for i in range(num_coeff):

