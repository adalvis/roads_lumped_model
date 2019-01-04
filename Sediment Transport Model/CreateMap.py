"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Map for Exner Solver test case
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.components import FastscapeEroder, FlowAccumulator, DepressionFinderAndRouter

from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.sans-serif'] = 'Arial Narrow'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


#%% Create grid and plot map

init = np.ones([100])
surface = (init*12) + np.random.rand(init.size)/100000.

mg_init = RasterModelGrid((10,10), spacing=(1,1))
z_init = mg_init.add_field('topographic__elevation', surface + mg_init.node_y*0.05 + mg_init.node_x*0.05, at = 'node')
dzdt_init = mg_init.add_zeros('node', 'erosion__rate')
qs_init = mg_init.add_zeros('node', 'sediment__discharge')

mg = RasterModelGrid((10,10), spacing=(1,1))
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05 + mg.node_x*0.05, at = 'node')
dzdt = mg.add_zeros('node', 'erosion__rate')
qs = mg.add_zeros('node', 'sediment__discharge')

qs_in = np.zeros(mg.number_of_nodes)


mg.set_closed_boundaries_at_grid_edges(True, True, True, True)

outlet_id = np.argmin(mg.at_node['topographic__elevation'])
mg.set_watershed_boundary_condition_outlet_id(outlet_id, mg.at_node['topographic__elevation'], 
                                              nodata_value=-9999.)


plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'out', bottom = True, 
               left = True, top = False, right = False)
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'jet')


#%% Create initial drainage network using FastScapeEroder


fr = FlowAccumulator(mg, flow_director='D8')
fsc = FastscapeEroder(mg, K_sp=.01, m_sp=2, n_sp=2)
df = DepressionFinderAndRouter(mg)

fsc_dt = 100.

for x in range(100):
    fr.run_one_step()
    df.map_depressions()
    flooded = np.where(df.flood_status == 3)[0]
    fsc.run_one_step(dt=fsc_dt, flooded_nodes=flooded)
    mg.at_node['topographic__elevation'][0] += 0.001 # Uplift


#%%
plt.figure()
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')


plt.figure()
ax2 = plt.axes(projection = '3d')
X = mg.node_x.reshape(10,10)
Y = mg.node_y.reshape(10,10)
Z = z.reshape(10,10)
ax2.plot_surface(X, Y, Z, cmap = 'jet')


