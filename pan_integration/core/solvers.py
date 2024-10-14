import torch
from torch import Tensor, tensor, nn, cos, pi, arange, hstack, cat, arccos, linspace
from torch.func import vmap, jacrev
from torch import linalg
from typing import Tuple


def T_grid(t, num_coeff_per_dim):
    num_points = len(t)
    out = torch.empty(num_coeff_per_dim, num_points, device=t.device)

    out[0, :] = torch.ones(num_points)
    out[1, :] = t

    for i in range(2, num_coeff_per_dim):
        out[i, :] = 2 * t * out[i - 1, :] - out[i - 2, :]

    return out


def U_grid(t, num_coeff_per_dim):
    num_points = len(t)
    out = torch.empty(num_coeff_per_dim, num_points, device=t.device)

    out[0, :] = torch.ones(num_points)
    out[1, :] = 2 * t

    for i in range(2, num_coeff_per_dim):
        out[i, :] = 2 * t * out[i - 1, :] - out[i - 2, :]

    return out


def DT_grid(t, num_coeff_per_dim):
    num_points = len(t)

    out = torch.vstack(
        [
            torch.zeros(1, num_points, dtype=torch.float, device=t.device),
            U_grid(t, num_coeff_per_dim - 1)
            * torch.arange(1, num_coeff_per_dim, dtype=torch.float, device=t.device)[
                :, None
            ],
        ]
    )

    return out


class PanSolver:
    def __init__(
        self,
        num_coeff_per_dim,
        device=None,
        callback=None,
        tol=0.01,
        patience=3,
        max_iters=100,
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.callback = callback
        self._num_points = num_coeff_per_dim - 2
        self.num_coeff_per_dim = num_coeff_per_dim

        self.tol = tol
        self.patience = patience

        self.max_iters = max_iters

    @property
    def num_points(self):
        return self._num_points

    @num_points.setter
    def num_points(self, value):
        (
            self.t_scld,
            self.PHI,
            self.DPHI,
            self.PHI_r,
            self.DPHI_r,
            self.PHI_r_inv,
            self.DPHI_r_inv,
        ) = self.calc_independent(self.num_coeff_per_dim, value, device=self.device)
        self._num_points = value

    @property
    def num_coeff_per_dim(self):
        return self._num_coeff_per_dim

    @num_coeff_per_dim.setter
    def num_coeff_per_dim(self, value):
        (
            self.t_scld,
            self.PHI,
            self.DPHI,
            self.PHI_r,
            self.DPHI_r,
            self.PHI_r_inv,
            self.DPHI_r_inv,
        ) = self.calc_independent(value, self.num_points, device=self.device)
        self._num_coeff_per_dim = value

    @staticmethod
    def calc_independent(num_coeff_per_dim, num_points, device):
        N = num_points
        # chebyshev nodes of the second kind
        k = arange(N)
        t_scld = -cos(pi * k / (N - 1)).to(device)

        # t_scld = torch.linspace(-1, 1, num_points)

        PHI = T_grid(t_scld, num_coeff_per_dim)
        DPHI = DT_grid(t_scld, num_coeff_per_dim)

        DPHI_r = DPHI[2:, 1:] - DPHI[2:, [0]]
        DPHI_r_inv = linalg.pinv(DPHI_r)

        PHI_r = (
            -PHI[2:, [0]]
            - DPHI[2:, [0]]
            - DPHI[2:, [0]] * t_scld[None, 1:]
            + PHI[2:, 1:]
        )
        PHI_r_inv = linalg.pinv(PHI_r)

        return t_scld, PHI, DPHI, PHI_r, DPHI_r, PHI_r_inv, DPHI_r_inv

    def _add_head(self, B_R, dt, y_init, f_init):
        Phi_R0 = self.PHI[2:, [0]]
        DPhi_R0 = self.DPHI[2:, [0]]

        phi_inv = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=self.device)
        b_01 = (
            torch.stack([y_init, (1 / dt) * f_init], dim=-1)
            - B_R @ torch.hstack([Phi_R0, DPhi_R0])
        ) @ phi_inv

        return torch.cat([b_01, B_R], dim=-1)


    def _gd(
        self,
        f,
        t_lims: Tuple,
        y_init: Tensor,
        f_init: Tensor,
        B_R: Tensor = None,
    ):
        dims = y_init.shape
        a =  (t_lims[1] - t_lims[0]) / 2
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_scld + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]
        B_R = torch.zeros(*dims, self.num_coeff_per_dim - 2, device=self.device, requires_grad=True)

        h = 1.0
        gamma = torch.tensor(self.gamma, device=self.device)

        C = a * f_init + y_init + a * f_init * self.t_scld[None, 1:]

        optimizer = torch.optim.SGD(params = [B_R], lr=1e-5)

        def loss_fn(B_R):
            y_k = C + B_R @ self.PHI_r

            f_k = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                t_true[1:],
                y_k,
            )

            Dt =  (2*a) / (self.num_points -1)

            Is = torch.cumsum( f_k, dim=-1 )  / Dt

            # print(f"{y_k}  \n  {Is} ")

            return torch.sum(( y_k -  Is)**2 )

        for i in range(self.max_iters):
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = loss_fn(B_R)
                loss.backward()
                optimizer.step()

            if self.callback is not None:
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                self.callback(i, t_lims, y_init, f_init, B)

        return None, None, None

    def _lin_prog(
        self,
        f,
        t_lims: Tuple,
        y_init: Tensor,
        f_init: Tensor,
        B_R: Tensor = None,
    ):
        dims = y_init.shape
        a = 0.5 * (t_lims[1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_scld + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]

        C = a * f_init + y_init + a * f_init * self.t_scld[None, 1:]

        if B_R is None:
            B_R = torch.zeros(*dims, self.num_coeff_per_dim - 2, device=self.device)

        h = 1.0
        gamma = torch.tensor(self.gamma, device=self.device)

        def g(B):
            y_k = C + B @ self.PHI_r

            f_k = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                t_true[1:],
                y_k,
            )
            return f_k, y_k

        prev_rel_error = torch.inf
        for i in range(1, self.max_iters):

            J, _ = torch.func.jacrev(g, has_aux=True)(B_R)


            if self.callback is not None:
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                self.callback(i, t_lims, y_init, f_init, B)

            # if torch.norm(rel_error) < self.tol:
            #     break


        # return y_k[..., -1], f_k[..., -1], B_R
        return 3*[None]

    def _zero_order(
        self,
        f,
        t_lims: Tuple,
        y_init: Tensor,
        f_init: Tensor,
        B_R: Tensor = None,
    ):
        dims = y_init.shape
        a = 0.5 * (t_lims[1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_scld + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]

        C = a * f_init + y_init + a * f_init * self.t_scld[None, 1:]

        if B_R is None:
            B_R = torch.zeros(*dims, self.num_coeff_per_dim - 2, device=self.device)

        prev_rel_error = torch.inf
        avg = 1
        for i in range(1, self.max_iters):

            y_k = C + B_R @ self.PHI_r

            f_k = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                t_true[1:],
                y_k,
            )

            Dy_k = B_R @ self.DPHI_r
            rel_error = torch.norm((a * f_k - a * f_init) - Dy_k)

            if rel_error < self.tol:
                return y_k[..., -1], f_k[..., -1], B_R

            avg += (torch.sign(rel_error - prev_rel_error) + 1)/ 2
            h = 1 / avg

            B_R = (1-h)*B_R + h* (a * f_k - a * f_init) @ self.DPHI_r_inv

            prev_rel_error = rel_error

            if self.callback is not None:
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                if self.callback(i, t_lims, y_init, f_init, B):
                    return y_k[..., -1], f_k[..., -1], B_R

        return y_k[..., -1], f_k[..., -1], B_R


    def _lstsq(
            self,
            f,
            t_lims: Tuple,
            y_init: Tensor,
            f_init: Tensor,
            B_R: Tensor = None,
    ):
        dims = y_init.shape
        a = 0.5 * (t_lims[1] - t_lims[0])
        t_true = t_lims[0] + 0.5 * (t_lims[1] - t_lims[0]) * (self.t_scld + 1)

        y_init = y_init[..., None]
        f_init = f_init[..., None]

        C = a * f_init + y_init + a * f_init * self.t_scld[None, 1:]

        if B_R is None:
            B_R = torch.randn(*dims, self.num_coeff_per_dim - 2, device=self.device)

        def g(B_R):

            y_k = C + B_R @ self.PHI_r

            f_k = vmap(f, in_dims=(0, -1), out_dims=(-1))(
                t_true[1:],
                y_k,
            )

            return f_k, (y_k, f_k)

        for i in range(1, self.max_iters):

            # naive version, no jvp
            J_k, (y_k, f_k) = jacrev(g, has_aux=True)(B_R)

            # TODO: figure out what to do with cross batch jacobian
            J_k = J_k.squeeze().mT[None]
            A = self.DPHI_r #- a*J_k
            q = B_R @ self.DPHI_r - a*f_k + a*f_init

            h = 0.01
            B_R = B_R + h * q @ A.mT @ linalg.inv(A @ A.mT)

            if self.callback is not None:
                B = self._add_head(
                    B_R.detach().clone(), 1 / a, y_init[..., 0], f_init[..., 0]
                )
                self.callback(i, t_lims, y_init, f_init, B)

        return y_k[..., -1], f_k[..., -1], B_R


    def solve(
        self,
        f,
        t_span,
        y_init,
        f_init=None,
        B_init=None,
        method='zero',
        **kwargs,
    ):

        func = None
        match method:
            case 'zero':
                func = self._zero_order
            case 'lstsq':
                func = self._lstsq

        if f_init is None:
            f_init = f(t_span[0], y_init)

        y_eval = [y_init]

        Bk = B_init
        yk = y_init
        fk = f_init
        for t_lims in zip(t_span, t_span[1:]):
            yk, fk, Bk = func(
                f,
                t_lims,
                yk,
                fk,
                Bk,
            )
            y_eval.append(yk)

        return torch.stack(y_eval, dim=0), None, Bk
