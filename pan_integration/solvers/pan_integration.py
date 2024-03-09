import torch
from torch import tensor, Tensor, cos, sin, ones, arange, arccos, sqrt, cosh, arccosh
from ..utils.plotting import VfPlotter, wait
from ..optim import newton
from torch import pi as PI


def cheby_pol(t, n, t0, t1):
    t_scl = 2 * (t - t0) / (t1 - t0) - 1
    return cos(n * arccos(t_scl))


def Dcheby_pol(t, n, t0, t1):
    t_scl = 2 * (t + t0) / (t1 - t0) - 1
    return 2 * n * sin(n * arccos(t_scl)) / ((t1 - t0) * sqrt(1 - t_scl ** 2))


def _phis_init_cond(t_lims, num_points, num_coeff_per_dim):
    t = torch.linspace(*t_lims, num_points)[1:-1]
    n = torch.arange(1, num_coeff_per_dim + 1, dtype=torch.float)

    ts, ns = torch.meshgrid(t, n, indexing='ij')

    _Phi_head = (-1) ** torch.arange(1, num_coeff_per_dim + 1)[None]
    _Phi_tail = torch.ones(1, num_coeff_per_dim, dtype=torch.float)
    _Phi = torch.vstack([_Phi_head, cheby_pol(ts, ns, *t_lims), _Phi_tail])
    _DPhi_head = torch.arange(1, num_coeff_per_dim + 1) ** 2 * (-1) ** torch.arange(num_coeff_per_dim)[None]
    _DPhi_tail = torch.arange(1, num_coeff_per_dim + 1, dtype=torch.float)[None] ** 2
    _DPhi = torch.vstack([_DPhi_head, Dcheby_pol(ts, ns, *t_lims), _DPhi_tail])

    return None


def _coarse_euler_lstsq(f, y_init, num_solver_steps, step, num_coeff_per_dim):
    dims = y_init.shape[0]
    step_size = step / (num_solver_steps - 1)

    y_eul = torch.zeros(num_solver_steps, dims)
    y_eul[0, :] = y_init
    f_y_eul = torch.zeros(num_solver_steps, dims)
    f_y_eul[0, :] = f(y_init)

    # forward euler
    for i in range(1, num_solver_steps):
        y_cur = y_eul[i - 1, :]
        f_y_eul[i, :] = f(y_cur)
        f_cur = f(y_cur)
        y_eul[i, :] = y_cur + step_size * f_cur

    # least squares to get Bs of euler
    Phi_c, Phi_s, Phi_s_1, DPhi_c, DPhi_s, DPhi_s_1 = _phis_init_cond(step, num_solver_steps, num_coeff_per_dim)
    PHI = torch.vstack([torch.hstack([Phi_c, Phi_s]),
                        torch.hstack([DPhi_c, DPhi_s])])

    Y = torch.vstack([y_eul, f_y_eul]) - torch.vstack([Phi_s_1, DPhi_s_1]) @ f_y_eul[:1, :] - torch.vstack(
        [torch.ones(num_solver_steps, 1), torch.zeros(num_solver_steps, 1)]) * y_init

    B_els = torch.linalg.lstsq(PHI, Y).solution

    return B_els.reshape(-1)


def pan_int(
        f,
        y_init: Tensor,
        t_lims: list,
        num_coeff_per_dim: int,
        num_points: int,
        step: float = None,
        etol=1e-3,
        callback: callable = None,
        plotter: VfPlotter = None
) -> Tensor:
    global COUNTER
    COUNTER = 0

    wait()

    if step is None:
        step = t_lims[1] - t_lims[0]

    dims = y_init.shape[0]
    cur_interval = [t_lims[0], t_lims[0] + step]

    approx = torch.tensor([])
    B = torch.rand((num_coeff_per_dim - 2) * dims)
    while True:

        if cur_interval[1] > t_lims[1]:
            cur_interval[1] = t_lims[1]
            step = cur_interval[1] - cur_interval[0]

        Phi_c, Phi_s, Phi_s_1, DPhi_c, DPhi_s, DPhi_s_1 = _phis_init_cond(t_lims, num_points, num_coeff_per_dim)

        def error_func(B_vec: Tensor) -> tuple:
            with torch.no_grad():
                global COUNTER
                COUNTER += 1
                print(COUNTER)

            # CONVERT VECTOR BACK TO MATRICES
            #  the first half of B comprises the Bc matrix minus the first row
            Bc = B_vec[:(num_coeff_per_dim // 2 - 1) * dims].reshape(num_coeff_per_dim // 2 - 1, dims)
            # the rest comprises the Bs matrix minus the first row
            Bs = B_vec[(num_coeff_per_dim // 2 - 1) * dims:].reshape(num_coeff_per_dim // 2 - 1, dims)

            approx = Phi_c @ Bc + Phi_s @ Bs + Phi_s_1 @ f(y_init)[None] + y_init
            Dapprox = DPhi_c @ Bc + DPhi_s @ Bs + DPhi_s_1 @ f(y_init)[None]

            error = torch.sum((f(approx) - Dapprox) ** 2) / (num_points * dims)

            return error, approx

        B = _coarse_euler_lstsq(f, y_init, 10, step, num_coeff_per_dim)
        COUNTER += 5

        B = newton(error_func, B, has_aux=True, tol=etol,
                   callback=lambda B: callback(B, y_init, cur_interval))

        # TODO: make newton return the aux of error_func instead of recalculating
        (error, local_approx) = error_func(B)
        COUNTER -= 1
        y_init = local_approx[-1, :]

        approx = torch.cat((approx, local_approx))
        if cur_interval[1] == t_lims[1]:
            print(f"PAN Integration took {COUNTER} function evaluations")
            return approx
        else:
            cur_interval = [cur_interval[1], cur_interval[1] + step]
