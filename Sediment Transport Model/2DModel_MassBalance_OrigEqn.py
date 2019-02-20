"""
Author: Amanda Manaster
Date: 10/26/2018
Purpose: Re-writing functions in straight script form for testing.

Updated: 12/20/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.components import FlowAccumulator, LinearDiffuser
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
init = np.ones([625])
np.random.seed(2)
surface = init + np.random.rand(init.size)/55.

mg = RasterModelGrid((25,25), 1)
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05, at = 'node')

mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

qs_in = np.zeros(mg.number_of_nodes)
qs_out = np.zeros(mg.number_of_nodes)
dqs_dx = np.zeros(mg.number_of_nodes)
dzdx = np.zeros(mg.number_of_nodes)

dzdt = np.zeros(mg.number_of_nodes)
dt = 0.1
T = np.arange(0, 1100, 50)


plt.figure(1)
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'out', bottom = True, 
               left = True, top = False, right = False)
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

fa = FlowAccumulator(mg, surface = 'topographic__elevation', flow_director = 'FlowDirectorD8',
                     depression_finder = 'DepressionFinderAndRouter', routing = 'D8')

fa.run_one_step()
da = mg.at_node['drainage_area']

ld = LinearDiffuser(mg, linear_diffusivity=0.005)

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

for l in range(len(T)):
    while t < T[l]:
        
        fa.run_one_step()
#        flooded_nodes = np.where(fa.depression_finder.flood_status == 3)[0]
        
        ld.run_one_step(dt)
        
        da = mg.at_node['drainage_area']
        
        dzdx = mg.calc_slope_at_node(z)
        
        src_nodes = mg.at_node['flow__upstream_node_order']
        dst_nodes = mg.at_node['flow__receiver_node']
        
        defined_flow_receivers = np.not_equal(mg.at_node["flow__link_to_receiver_node"], -1)
        flow_link_lengths = mg.length_of_d8[mg.at_node["flow__link_to_receiver_node"]]
          
        for i in range(mg.number_of_nodes):
            src_id = src_nodes[i]
            dst_id = dst_nodes[src_id]
            
            qs_out[src_id] = k_t * da[src_id]**m * dzdx[src_id]**n
            
            if mg.cell_area_at_node[src_id] == 0:
                dqs_dx[src_id] = 0
            else:
                dqs_dx[src_id] = (qs_out[src_id] - qs_in[src_id]) / mg.cell_area_at_node[src_id]
            
#            if flooded_nodes is not None:
#                dqs_dx[flooded_nodes] = 0.
#            else:
#                reversed_flow = z < z[dst_id]
#                dqs_dx[reversed_flow] = 0.
                
            dzdt[src_id] = -dqs_dx[src_id]    

            qs_in[dst_id] = qs_out[src_id]
            
            if z[src_id] < z[dst_id]:
                z[src_id] = z[dst_id]*1.00001
            
        z0 = z.copy()
        z = z0 + dzdt*dt
            
                
        t += dt
        print(t)

#    plt.figure()
#    drainage_plot(mg, 'drainage_area')
#
#    plt.figure()
#    imshow_grid(mg, z, plot_name = '%i years' % T[l], var_name = 'Elevation', 
#                var_units = 'm', grid_units = ('m','m'), cmap = 'gist_earth', vmin = 1.0, vmax = 2.2)

    plt.figure()
    ax2 = plt.axes(projection = '3d')
    X = mg.node_x.reshape(mg.shape)
    Y = mg.node_y.reshape(mg.shape)
    Z = z.reshape(mg.shape)
    plt.title('%i years' % T[l])
    ax2.set_xlabel('X (m)', fontsize = 10)
    ax2.set_ylabel('Y (m)', fontsize = 10)
    ax2.set_zlabel('Elevation (m)', fontsize = 10)
    ax2.plot_surface(X, Y, Z, cmap = 'gist_earth')
#    plt.savefig('C:/Users/Amanda/Desktop/3DFigs/year%i.png' % T[l])

    plt.figure()
    plt.title('Slope-Area Plot for Initial Conditions')
    plt.xlabel('Area (m$^2$)')
    plt.ylabel('Slope (-)')
    plt.loglog(drainage_area, slope[mg.core_nodes], 'b-')
    plt.loglog(da[mg.core_nodes], dzdx[mg.core_nodes], 'ko')
    plt.show()


#for keeping track of how long the model runs
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) 