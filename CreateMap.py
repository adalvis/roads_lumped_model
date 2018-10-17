"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Map for Exner Solver test case
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


#%% Create grid and plot map

init = np.ones([10000])
surface = (init*12) + np.random.rand(init.size)/100000.


mg = RasterModelGrid(100, 100, spacing=(1,1))
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05 + mg.node_x*0.05, at = 'node')
qs = mg.add_zeros('node', 'sediment__discharge')

#mg.set_fixed_value_boundaries_at_grid_edges(True, True, True, True)
mg.set_closed_boundaries_at_grid_edges(True, True, True, True) 

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'out', bottom = 'on', 
               left = 'on', top = 'off', right = 'off')
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'jet')