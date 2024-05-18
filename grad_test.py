import torch
from torch import nn
from pan_integration.numerics.functional import pan_int, make_pan_adjoint
from torchdyn.models import NeuralODE

torch.manual_seed(42)


class NN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand(5, 5))
        self.w2 = torch.nn.Parameter(torch.rand(5, 5))
        self.w3 = torch.nn.Parameter(torch.rand(5, 5))
        self.nfe = 0

    def forward(self, t, y, *args, **kwargs):
        self.nfe += 1
        return torch.tanh( self.w3 @ torch.tanh(self.w2 @ torch.tanh(self.w1 @y)))


class PanODE(nn.Module):
    def __init__(
        self,
        vf,
        num_coeff_per_dim,
        num_points,
        tol_zero=1e-3,
        tol_one=1e-5,
        max_iters_zero=10,
        max_iters_one=10,
        optimizer_class=None,
        optimizer_params=None,
        init="random",
        coarse_steps=5,
        callback=None,
        metrics=True,
    ):
        super().__init__()

        self.vf = vf
        self.thetas = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        self.pan_int = make_pan_adjoint(
            self.vf,
            self.thetas,
            num_coeff_per_dim,
            num_points,
            tol_zero=tol_zero,
            tol_one=tol_one,
            max_iters_zero=max_iters_zero,
            max_iters_one=max_iters_one,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            init=init,
            coarse_steps=coarse_steps,
            callback=callback,
            metrics=metrics,
        )

    def forward(self, y_init, t, *args, **kwargs):
        t_eval, traj, metrics = self.pan_int(y_init, t)
        return t_eval, traj, metrics


if __name__ == "__main__":
    vf = NN()
    solver_args = dict(
        num_coeff_per_dim=64,
        num_points=64,
        tol_zero=1e-3,
        tol_one=1e-4,
        max_iters_zero=30,
        max_iters_one=0,
        optimizer_class=torch.optim.SGD,
        optimizer_params=dict(lr=1e-9),
        init='random',
        coarse_steps=5,
        metrics=True,
    )
    y_init = torch.rand(1, 5, 5)
    t_span = torch.linspace(0, 1, 2)

    pan_ode_model = PanODE(vf, **solver_args)
    _, traj_pan, _ = pan_ode_model(y_init, t_span)
    L_pan = torch.sum((traj_pan[-1] -1*torch.ones(5, 5)) ** 2)
    L_pan.backward()
    grads_pan = [w.grad for w in vf.parameters()]

    vf.zero_grad()

    ode_model = NeuralODE(
        vf, sensitivity="adjoint", return_t_eval=False, atol=1e-9, atol_adjoint=1e-9
    )
    traj = ode_model(y_init, t_span)
    L = torch.sum((traj[-1] - 1*torch.ones(5, 5)) ** 2)
    L.backward()
    grads = [w.grad for w in vf.parameters()]

    print('SOLUTION \n')
    print(traj[-1], '\n', traj_pan[-1], '\n')

    print('GRADS\n')
    print(grads[0], '\n', grads_pan[0], '\n')
    print(grads[1],'\n',grads_pan[1])

    # print(grads_pan[2]/ grads[2])
