import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import cond
from scipy.linalg import solve
from scipy import interpolate
import math

mpl.use('TkAgg')
plt.close('all')

PI = np.pi
XLIMS = (0, 1)
YLIMS = (0, 1)

#########
# REMEZ #
#########
alpha = 0.5
beta = 1


def f(t: np.ndarray, y: np.ndarray, alpha= alpha, beta=beta) -> np.ndarray:
    return beta * sin(alpha * PI * y * t)


def partial_f(t: np.ndarray, y: np.ndarray, alpha= alpha, beta=beta) -> np.ndarray:
    return beta * alpha * PI * t * cos(alpha * PI * y * t)


ts = np.linspace(-0.1, 1.1, 40)
ys = np.linspace(YLIMS[0], YLIMS[1], 40)
Ts, Ys = np.meshgrid(ts, ys)
slopes = f(Ts, Ys)

y_0 = np.random.rand(1)
solver_solution = solve_ivp(f, [YLIMS[0], YLIMS[1]], y_0, method='LSODA', atol=1e-9)
real_traj, real_t = solver_solution.y.squeeze(), solver_solution.t

true_gradient = f(real_t, real_traj)



# f(t)
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("Slope field")
ax2.set_title("$\delta(t)$")
ax3.set_title("Derivatives")
ax1.set_xlim(-0.1, 1.1)
ax2.set_xlim(XLIMS[0]-0.1, XLIMS[1]+0.1)
ax3.set_xlim(XLIMS[0]-0.1, XLIMS[1]+0.1)
ax1.set_ylim(YLIMS[0], YLIMS[1])

stream_kwargs = {
   'linewidth' : 1.1
}

ax1.streamplot(Ts, Ys, np.ones_like(slopes), slopes, **stream_kwargs)
ax1.plot(real_t, real_traj, 'r-', linewidth='2')
ax2.plot([XLIMS[0], XLIMS[1]], [0, 0], color="black", linewidth=0.5)
ax3.plot(real_t, true_gradient, 'r-', label="Ground Truth")
ax3.plot([XLIMS[0], XLIMS[1]], [0, 0], color="black", linewidth=0.5)


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
            pairs = np.convolve(np.array([1, 1]), np.abs(delta[indices]) , mode='valid')
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

    return  np.delete(all_extrema_idx, np.array(indices_to_be_deleted, dtype=np.int32))


def neural_remez(y_0: np.ndarray, num_params: int = 10, max_iters: int = 1, dense_sampling: int = 1000) -> np.ndarray:
    iters = 0

    delta: np.ndarray = np.array((dense_sampling, 1))
    t_dense = np.linspace(XLIMS[0], XLIMS[1], dense_sampling).reshape(dense_sampling, 1)

    ones_alter = np.empty((num_params + 1, 1), float)
    ones_alter[::2, 0] = -1
    ones_alter[1::2, 0] = 1

    #
    n: np.ndarray = np.arange(1, num_params+1)
    # Φ_c
    PHIc = lambda t: 2*sin(n * PI * t * .5)**2
    # Φ_s
    PHIs = lambda t: -PI * n * sin(n * PI * t)

    a: np.ndarray = np.random.rand(num_params, 1) / np.arange(1, num_params+1).reshape(num_params,1)
    a[-1] = -10

    # plotting
    ax1.plot([XLIMS[0]], [y_0], 'go') # init point
    art_state_approx, = ax1.plot([], [], 'g--')
    art_delta, = ax2.plot([], [])
    art_all_extr, = ax2.plot([], [], 'o', color='red', markersize=6)
    art_alter_extr, = ax2.plot([], [], 'o', color='orange', markersize=7)
    art_final_extr, = ax2.plot([], [], 'o', color='green', markersize=8)
    art_f_y0, = ax3.plot([], [], color='orange', label="$f(t, y_0)$")
    art_phiphi, = ax3.plot([], [], color='goldenrod',linestyle='dashed', linewidth=1.2, label="$\Phi_s - \partial f \cdot \Phi_c$")
    ax3.legend()

    while iters < max_iters:
        print(iters, a)

        # CALCULATE THE DIFFERENCE BETWEEN TRUE GRADIENT AND THE GRADIENT OF THE APPROXIMATION
        gradient_approx = PHIs(t_dense) @ a
        state_approx = y_0 - PHIc(t_dense) @ a
        partial_f_dense = partial_f(t_dense, y_0)
        # delta =  f(t_dense, state_approx) - gradient_approx
        phis_diff = (PHIs(t_dense) - partial_f_dense * PHIs(t_dense) ) @ a
        f_y0 = f(t_dense, y_0)
        delta = phis_diff - f_y0
        delta = np.squeeze(delta)

        art_delta.set_data(t_dense, delta)  # plot delta curve
        ax2.set_ylim((np.min(delta) * 1.1, np.max(delta) * 1.1))
        art_state_approx.set_data(t_dense, state_approx)
        art_f_y0.set_data(t_dense, f_y0)
        art_phiphi.set_data(t_dense, phis_diff)

        # FIND NUM_PARAMS+1 ALTERNATING MAXIMUMS
        # get all extrema
        all_extrema_idx = (np.diff(np.sign(np.diff(delta, append=0)), prepend=0) != 0).nonzero()[0]
        # remove first element (0)
        all_extrema_idx = np.concatenate((all_extrema_idx[1:], np.array([dense_sampling-1])))
        art_all_extr.set_data(t_dense[all_extrema_idx], delta[all_extrema_idx])

        # remove non-alternating
        alter_extrema_idx = keep_area_max(delta, all_extrema_idx)
        art_alter_extr.set_data(t_dense[alter_extrema_idx], delta[alter_extrema_idx])  # plot all extrema that alternate

        # keep N+1 extrema that are maximum and alternate
        if iters == 0 and alter_extrema_idx.size < num_params+1:
            # keep N+1 extrema that are maximum and alternate
            final_extrema_idx = np.arange(1 , dense_sampling,  dense_sampling / math.floor(num_params+1) ).astype(int)
        else:
            final_extrema_idx = keep_extrema(delta, alter_extrema_idx, num_params)

        art_final_extr.set_data(t_dense[final_extrema_idx], delta[final_extrema_idx])  # plot N+1 extrema to keep

        t_final = t_dense[final_extrema_idx]

        # CALCULATE NEW a
        partial_f_y0 = partial_f(t_final, y_0)# column vector, each element multiply with one row of PHIc
        # matrix X a
        Alpha = PHIs(t_final) - partial_f_y0 * PHIs(t_final)
        #                                    ^  broadcast not matmul!
        Alpha = np.hstack( (Alpha, ones_alter) )
        beta = f(t_final, y_0)
        a = solve(Alpha, beta)
        # discard last element (δ)
        a = a[:-1]

        fig1.suptitle(iters)
        # plots
        while True:
            if plt.waitforbuttonpress():
                break

        iters = iters + 1

    return a


neural_remez(y_0, max_iters=100, num_params=20)

