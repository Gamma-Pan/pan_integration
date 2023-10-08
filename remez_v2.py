import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import solve
import math
from typing import Callable

mpl.use('TkAgg')
plt.close('all')

PI = np.pi
XLIMS = (0, 1)
YLIMS = (0, 1)
TLIMS = (0, 1)

#########
# REMEZ #
#########
mu = 0.5
nu = 1


def spiral_0(y0: np.ndarray) -> np.ndarray:
    # return cos(PI * mu * y0) ** 2 + 0.1
    return 1 / (1 + np.exp(-y0))


def partial_0(y0: np.ndarray) -> np.ndarray:
    # return - sin(2 * PI * mu * y0) * PI * mu
    return (1-spiral_0(y0))*spiral_0(y0)


def spiral_1(y1: np.ndarray) -> np.ndarray:
    return sin(PI * nu * y1) + 0.1


def partial_1(y1: np.ndarray) -> np.ndarray:
    return cos(PI * nu * y1) * PI * nu


def spiral(y0: np.ndarray, y1: np.ndarray) -> tuple:
    dy0 = spiral_0(y0)
    dy1 = spiral_1(y1)
    return dy0, dy1


def f(t, y):
    return spiral(y[0], y[1])


y_init = np.random.rand(2) * .2
real_solution = solve_ivp(f, [TLIMS[0], TLIMS[1]], y_init, 'LSODA', np.linspace(TLIMS[0], TLIMS[1], 100), vectorized=True)
solution_t = real_solution.t
solution_y = real_solution.y
solution_y0 = solution_y[0,:]
solution_y1 = solution_y[1,:]

# spiral
fig_spiral, ax_spiral = plt.subplots()
ax_spiral.set_xlabel = '$y_0$'
ax_spiral.set_ylabel = '$y_1$'

lin_y0 = np.linspace(XLIMS[0], XLIMS[1])
lin_y1 = np.linspace(YLIMS[0], YLIMS[1])

Y0, Y1 = np.meshgrid(lin_y0, lin_y1)
DY0, DY1 = spiral(Y0, Y1)

stream_kwargs = {
    'linewidth': 1.1
}
ax_spiral.streamplot(Y0, Y1, DY0, DY1, **stream_kwargs)
ax_spiral.plot([y_init[0]], [y_init[1]], 'ro')
ax_spiral.plot(solution_y0, solution_y1, 'r-', label="solver trajectory")
ax_spiral.legend()
plt.show()


def keep_extrema(delta: np.ndarray, extrema_indices: np.ndarray, num_params: int) -> np.ndarray:
    def recursion(indices: np.ndarray):
        print(indices.size)
        # if indices are the same size as num parameters
        if indices.size == num_params + 1:
            return indices
        elif indices.size == num_params + 2:
            if abs(delta[indices[0]]) >= (delta[indices[-1]]):
                return np.delete(indices, -1)
            else:
                return np.delete(indices, 0)
        else:
            # convolve abs of extrema to find pairs
            pairs = np.convolve(np.array([1, 1]), np.abs(delta[indices]), mode='valid')
            # also calculate the sum of abs of last and first deltas
            edges = np.abs(delta[indices[0]]) + np.abs(delta[indices[-1]])

            # find index of smallest pair
            max_pair_index = np.argmin(pairs)
            # either remove indexes of pair and repeat
            if pairs[max_pair_index] < edges:
                return recursion(np.delete(indices, np.array([max_pair_index, max_pair_index + 1])))
            # or remove first and last and repeat
            else:
                return recursion(np.delete(indices, np.array([0, indices.size - 1])))

    return recursion(extrema_indices)


def keep_area_max(delta: np.ndarray, all_extrema_idx: np.ndarray):
    all_extrema = delta[all_extrema_idx]
    # only keep the extremum with the maximum abs value between two roots
    # 1. find which extrema have non-alternating signs
    # 1.1 using forward and backward differences to catch first and last indices of non-alter areas
    diff_f = np.diff(np.sign(all_extrema), prepend=0)
    diff_b = np.diff(np.sign(all_extrema), append=0)
    diff = diff_f * diff_b

    # 2. for the indices of not-alternating sign only keep the largest value
    # 2.1 find limits of changing sign areas
    nonalter_lims_left = ((diff_f == 0) ^ (diff == 0)).nonzero()
    nonalter_lims_right = ((diff_b == 0) ^ (diff == 0)).nonzero()

    # 2.2 replace all indices between lims with argmax
    indices_to_be_deleted = []
    for i, j in zip(nonalter_lims_left[0], nonalter_lims_right[0]):
        idx = np.argmax(np.abs(all_extrema[i:j + 1]))
        all_area_idxs = np.delete(np.arange(i, j + 1), idx)
        indices_to_be_deleted = np.concatenate((indices_to_be_deleted, all_area_idxs))

    return np.delete(all_extrema_idx, np.array(indices_to_be_deleted, dtype=np.int32))


def neural_remez(f: Callable,partial_f: Callable, y_init: np.ndarray, num_params: int = 10, max_iters: int = 1, dense_sampling: int = 1000) -> np.ndarray:
    iters = 0

    delta: np.ndarray = np.array((dense_sampling, 1))
    t_dense = np.linspace(TLIMS[0], TLIMS[1], dense_sampling).reshape(dense_sampling, 1)

    ones_alter = np.empty((num_params + 1, 1), float)
    ones_alter[::2, 0] = -1
    ones_alter[1::2, 0] = 1

    # num_params
    n: np.ndarray = np.arange(1, num_params + 1)
    # Φ_c
    PHIc = lambda t: - 2*sin(n * PI * t * .5) ** 2
    # Φ_s
    PHIs = lambda t: PI * n * sin(n * PI * t)
    # Φ_I
    PHII = lambda t:  sin(n * PI * t) / (n*PI) - t
    # df/dy(y_init)
    dfdy = partial_f(y_init)

    a: np.ndarray = - np.random.rand(num_params, 1)

    # plotting
    fig_remez, (ax_time, ax_delta, ax_grads) = plt.subplots(1, 3)
    ax_time.set_title('Time domain')
    ax_delta.set_title('$\delta(t)$')
    ax_grads.set_title('Gradient domain')
    ax_time.plot(solution_t, solution_y0, 'r-')
    art_time_approx, = ax_time.plot([],[], 'g--')

    ax_delta.plot([TLIMS[0], TLIMS[1]], [0, 0], 'k-', linewidth=0.5)
    ax_delta.set_xlim( (TLIMS[0], TLIMS[1]) )
    art_delta, = ax_delta.plot([], [], color='limegreen')
    art_extr_all, = ax_delta.plot([], [], 'o', color='red', markersize=6)
    art_extr_alter, = ax_delta.plot([], [], 'o', color='orange', markersize=7)
    art_extr_final, = ax_delta.plot([], [], 'o', color='green', markersize=8)

    ax_grads.plot(solution_t, f(solution_y0), 'r-', label="true gradient")
    art_fy0t, = ax_grads.plot([], [], 'b-', label='$f(y0)*t$')
    art_phi, = ax_grads.plot([], [], color='teal', linestyle='--', label='$\Phi$')
    ax_grads.legend()

    while iters < max_iters:
        print(iters, a)

        # CALCULATE DELTA AND APPROXIMATIONS
        time_approx = y_init + PHIc(t_dense) @ a
        grad_approx = PHIs(t_dense) @ a
        delta = (PHII(t_dense) - dfdy * PHII(t_dense)) @ a - f(y_init)*t_dense
        #                                                     ^ scalar times vector
        delta = np.squeeze(delta)

        art_time_approx.set_data(t_dense, time_approx)

        art_delta.set_data(t_dense, delta)  # plot delta
        ax_delta.set_ylim(np.min(delta) * 1.1, np.max(delta)*1.1)

        art_fy0t.set_data(t_dense, f(y_init)*t_dense)
        art_phi.set_data(t_dense, (PHII(t_dense) - dfdy * PHII(t_dense)) @ a)
        ax_grads.set_ylim(np.min((PHII(t_dense) - dfdy * PHII(t_dense)) @ a) * 1.1, np.max((PHII(t_dense) - dfdy * PHII(t_dense)) @ a)*1.1)

        while True:
            if plt.waitforbuttonpress():
                break

        # FIND NUM_PARAMS+1 ALTERNATING MAXIMUMS
        # get all extrema
        all_extrema_idx = (np.diff(np.sign(np.diff(delta, append=0)), prepend=0) != 0).nonzero()[0]
        # remove first element (0)
        all_extrema_idx = np.concatenate((all_extrema_idx[1:], np.array([dense_sampling - 1])))
        art_extr_all.set_data(t_dense[all_extrema_idx], delta[all_extrema_idx])

        # remove non-alternating
        alter_extrema_idx = keep_area_max(delta, all_extrema_idx)
        art_extr_alter.set_data(t_dense[alter_extrema_idx], delta[alter_extrema_idx])  # plot all extrema that alternate

        # keep N+1 extrema that are maximum and alternate
        # if iters == 0 and alter_extrema_idx.size < num_params + 1:
        if  alter_extrema_idx.size < num_params + 1:
            # keep N+1 extrema that are maximum and alternate
            final_extrema_idx = np.arange(1, dense_sampling, dense_sampling / math.floor(num_params + 1)).astype(int)
        else:
            final_extrema_idx = keep_extrema(delta, alter_extrema_idx, num_params)

        art_extr_final.set_data(t_dense[final_extrema_idx], delta[final_extrema_idx])  # plot N+1 extrema to keep

        t_final = t_dense[final_extrema_idx]

        # CALCULATE NEW a
        # matrix X a
        Alpha = PHIc(t_final) - dfdy * PHII(t_final)
        #                                    ^  broadcast not matmul!
        Alpha = np.hstack((Alpha, ones_alter))
        beta = f(y_init) * t_final
        a = solve(Alpha, beta)
        # discard last element (δ)
        a = a[:-1]

        fig_remez.suptitle(iters)
        # plots
        while True:
            if plt.waitforbuttonpress():
                break

        iters = iters + 1

    return a

print(neural_remez(spiral_0, partial_0, y_init[0], max_iters=100, num_params=20))
