"""
Author: Amanda Manaster
Date: 10/26/2018
Purpose: Re-writing functions in straight script form for testing.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.components import FlowAccumulator
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
surface = (init*12) + np.random.rand(init.size)/100000.

mg_init = RasterModelGrid((10,10), 1)
z_init = mg_init.add_field('topographic__elevation', surface + mg_init.node_y*0.05 + mg_init.node_x*0.05, at = 'node')

mg = RasterModelGrid((10,10), 1)
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05 + mg.node_x*0.05, at = 'node')

qs_in = np.zeros(mg.number_of_nodes)
qs_out = np.zeros(mg.number_of_nodes)
dqs = np.zeros(mg.number_of_nodes)
dqs_dx = np.zeros(mg.number_of_nodes)

dzdt = np.zeros(mg.number_of_nodes)
dt = 0.0001

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'out', bottom = True, 
               left = True, top = False, right = False)
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

fa = FlowAccumulator(mg, surface = 'topographic__elevation', flow_director = 'FlowDirectorD8',
                     depression_finder = 'DepressionFinderAndRouter', routing = 'D8')

fa.run_one_step()

#%% Calculate slope at each node - initial, analytical
n = 2.
m = 2.
U = 5e-5
k_t = 0.01

slope = np.zeros(mg.number_of_nodes)

for node in mg.core_nodes:
    slope[node] = ((U/k_t)**(1/n)) * (mg.at_node['drainage_area'][node])**((1-m)/n)

drainage_area = mg.at_node['drainage_area'][mg.core_nodes]
drainage_area = drainage_area.copy()

#%% Calculate sediment discharge, divergence, and elevation change     
t = 0
while t < 100:

    fa.run_one_step()
    flooded = np.where(fa.depression_finder.flood_status == 3)[0]
    
    da = mg.at_node['drainage_area']
    
    ordered_nodes = np.flipud(mg.at_node['flow__upstream_node_order'])
    
    dzdx = np.tan(mg.calc_slope_at_node(z))
    
    flow_receiver = mg.at_node['flow__receiver_node']
    
    for i in range(mg.number_of_nodes):
        node = ordered_nodes[i]
        
        qs_out[node] = k_t * da[node]**m * dzdx[node]**n
        
        if mg.cell_area_at_node[node] == 0:
            dqs_dx[node] = 0
        else:
            dqs_dx[node] = (qs_out[node] - qs_in[node])/ mg.cell_area_at_node[node]
    
        qs_out[node] = qs_in[node] + dqs_dx[node]*mg.cell_area_at_node[node]

        if flooded is not None:
            qs_out[flooded] = 0.
        else:
            reversed_flow = z < z[flow_receiver]
            qs_out[reversed_flow] = 0.  
        
        qs_in[flow_receiver[node]] = qs_out[node]

    z0 = z.copy()
    z = z0 - dqs_dx*dt

    
    t += dt
    print(t)


plt.figure()
drainage_plot(mg, 'drainage_area')

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


#%%
plt.figure()
plt.title('Slope-Area Plot for Initial Conditions')
plt.xlabel('Area (m$^2$)')
plt.ylabel('Slope (-)')
plt.loglog(drainage_area, slope[mg.core_nodes], 'b-')
plt.loglog(da, dzdx, 'ko')
plt.show()

print(sum(z-z_init))

#for keeping track of how long the model runs
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) 