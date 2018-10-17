"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Test defined functions and calculate values @ t = 1
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

#%% Plot drainage area and flow accumulation initially
ordered_nodes, mg, da, q = ordered(mg)
ordered_desc = ordered_nodes[::-1]
plt.figure()
drainage_plot(mg, 'drainage_area')

#%% Initial conditions slope-area plot

dzdx, mg = calculate_slope(mg, z)

plt.figure()
plt.title('Slope-Area Plot for Initial Conditions')
plt.xlabel('Area (m$^2$)')
plt.ylabel('Slope (-)')
plt.loglog(mg.at_node['drainage_area'], dzdx, 'ko')
plt.xlim(10**(0)-10, 10**(4) + 5000)   
plt.ylim(10**(-4), 10**(-1)) 
plt.show()


#%% Calculate initial sediment discharge and plot

qs, mg = sed_disch(qs, mg, ordered_desc, da, dzdx)

plt.figure()
imshow_grid(mg, qs, plot_name = 'Sediment Discharge at t = 0', var_name = 'Sediment Discharge', 
            var_units = r'$\frac{Q^3}{s}$', grid_units = ('m','m'), cmap = 'jet')


#%% Calculate initial sediment divergence and plot

dqs_dx = div_qs(qs, mg)

plt.figure()
imshow_grid(mg, dqs_dx, plot_name = 'Divergence of Sediment Discharge at t = 0', var_name = 'Sediment Flux', 
            var_units = r'$\frac{Q^2}{s}$', grid_units = ('m','m'), cmap = 'jet')

#%% Solve for elevation change for 1 time step

mg = ExnerSolver(mg, ordered_desc, dqs_dx, 1)

plt.figure()
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')