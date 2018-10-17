"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Run model until equilibrium is reached
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from landlab.components import FlowAccumulator, FlowDirectorD8
from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#%% Create model driver

U = 5e-5
UA = U * da
UA[0] = 0
t = 0
dt = 1


#%%

while qs.all() >= UA.all():
    qs, mg = sed_disch(qs, mg, ordered_desc, da, dzdx)
    dqs_dx = div_qs(qs, mg)
    mg = ExnerSolver(mg, ordered_desc, dqs_dx, dt)
    
    ordered_nodes, mg, da, q = ordered(mg)
    ordered_desc = ordered_nodes[::-1]
    
    dzdx, mg = calculate_slope(mg, z)
    
    t += dt
    print(da)
    
    
plt.figure()
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

