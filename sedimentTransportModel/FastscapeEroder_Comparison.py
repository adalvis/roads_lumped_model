"""
Author: Amanda Manaster
Date: 2019-01-22
Purpose: How does FastscapeEroder compare to 2DModel_MassBalance_OrigEqn?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.components import FlowAccumulator, FastscapeEroder
from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#for keeping track of how long the model runs
from datetime import datetime
start_time = datetime.now()


#%% Create grid and plot map
init = np.ones([100])
np.random.seed(2)
surface = init + np.random.rand(init.size)/55.

mg = RasterModelGrid((10,10), 1)
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05, at = 'node')

mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

#mg.set_watershed_boundary_condition_outlet_id(12, z)


qs_in = np.zeros(mg.number_of_nodes)
qs_out = np.zeros(mg.number_of_nodes)
dqs_dx = np.zeros(mg.number_of_nodes)

dzdt = np.zeros(mg.number_of_nodes)
dt = 0.1
T = np.array([0, 50, 100, 500, 1000])


plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'out', bottom = True, 
               left = True, top = False, right = False)
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

fa = FlowAccumulator(mg, surface = 'topographic__elevation', flow_director = 'FlowDirectorD8',
                     depression_finder = 'DepressionFinderAndRouter', routing = 'D8')
fsc = FastscapeEroder(mg, K_sp = 0.01, m_sp = 2, n_sp = 2)


#%%
t = 0
for l in range(len(T)):
    while t < T[l]:

        fa.run_one_step()
        fsc.run_one_step(dt = dt)
        
        t += dt
        print(t)
        
#%%
    plt.figure()
    imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
               var_units = 'm', grid_units = ('m','m'), cmap = 'jet')
    
    plt.figure()
    drainage_plot(mg, 'drainage_area')

    plt.figure()
    ax2 = plt.axes(projection = '3d')
    X = mg.node_x.reshape(mg.shape)
    Y = mg.node_y.reshape(mg.shape)
    Z = z.reshape(mg.shape)
    plt.title('500 years of evolution')
    ax2.set_xlabel('X (m)', fontsize = 10)
    ax2.set_ylabel('Y (m)', fontsize = 10)
    ax2.set_zlabel('Elevation (m)', fontsize = 10)
    ax2.plot_surface(X, Y, Z, cmap = 'jet')


#for keeping track of how long the model runs
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) 