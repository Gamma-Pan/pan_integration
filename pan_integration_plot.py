import torch
from torch import linspace, meshgrid

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

grid_def = (100,100)
y1_lims = [0,1]
y2_lims = [0,1]

fig,ax = plt.subplots()
y1s = linspace(y1_lims[0],y1_lims[1], grid_def[0])
y2s = linspace(y2_lims[0],y2_lims[1], grid_def[1])

y1_grid, y_2grid = meshgrid(y1s, y2s, indexing="xy")

plt.show()