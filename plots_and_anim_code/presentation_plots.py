import torch
from torch import pi as PI, log, linalg, sin
from torch import cos, linspace, arange, randn
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from pan_integration.core.solvers import T_grid, DT_grid
# %%

plt.close("all")

num_points = 11
t = linspace(-1, 1, num_points)
n = randn(num_points) / 10
y = abs(t)

plt.plot(t, y, 'o', zorder=2, color = 'blue')

num_coeff = num_points
phi = T_grid(t, num_coeff)
b = linalg.lstsq(phi.T, y).solution[None]

t_plot = linspace(-1, 1, 100)
phi_plot = T_grid(t_plot, num_coeff)
approx = b@phi_plot
plt.plot(t_plot, approx[0], zorder=1, color='cyan')
plt.show()

# %%
plt.figure()
k = arange(num_points)
t_cheb = -cos( k*PI / (num_points-1) )
y_cheb = abs(t_cheb)
plt.plot(t_cheb, y_cheb, 'o', zorder=2, color='darkgreen')

phi_cheb = T_grid(t_cheb, num_coeff)
b = linalg.lstsq(phi_cheb.T, y_cheb).solution[None]

approx = b@phi_plot
plt.plot(t_plot, approx[0], zorder=1, color='springgreen')
plt.show()


# %%

















