"""
Created on Wed Feb 20 12:05:00 2019

Author: Amanda
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.components import FlowAccumulator
from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#%% 2D surface elevation plot of road surface

mg = RasterModelGrid(355,47,0.225) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
init_elev = np.zeros(mg.number_of_nodes, dtype = float) #initialize the elevation grid

np.random.seed(2)
surface = init_elev + np.random.rand(init_elev.size)/25



#%%
z = mg.add_zeros('topographic__elevation', at = 'node') #create the topographic__elevation field

road_peak = 16 #peak crowning height occurs at this x-location
up = 0.0067 #rise of slope from ditchline to crown
down = 0.0035 #rise of slope from crown to hillslope

for i in range(0,355): #loop through road length
    elev = 0 #initialize elevation placeholder
    
    for j in range(0, 47): #loop through road width
        
        
        if j < road_peak: #update latitudinal slopes based on location related to road_peak
            elev += up
        else:
            elev -= down
            
        z[i*47 + j] = elev #update elevation based on x & y locations    

z = z + mg.node_y*0.05 + surface #add longitudinal slope to road segment

mg.set_closed_boundaries_at_grid_edges(False, True, False, True)
    
plt.figure(figsize = (4,10))
imshow_grid(mg, z, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_0.05.png', bbox_inches = 'tight')
plt.show()

#%%
qs_in = np.zeros(mg.number_of_nodes)
qs_out = np.zeros(mg.number_of_nodes)
dqs_dx = np.zeros(mg.number_of_nodes)

dzdt = np.zeros(mg.number_of_nodes)
dt = 0.1
T = np.array([0,1])


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
for l in range(len(T)):
    while t < T[l]:

        fa.run_one_step()
        
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
                
            dzdt[src_id] = -dqs_dx[src_id]    

            qs_in[dst_id] = qs_out[src_id]
            
            if z[src_id] < z[dst_id]:
                z[src_id] = z[dst_id]*1.00001
            
        z0 = z.copy()
        z = z0 + dzdt*dt
            
                
        t += dt
        print(t)
    
#%%
plt.figure()
drainage_plot(mg, 'drainage_area')

plt.figure()
imshow_grid(mg, dzdx, plot_name = 'Slope Plot', var_name = 'Slope', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

plt.figure()
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

plt.figure()
ax2 = plt.axes(projection = '3d')
X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)
plt.title('1 year of evolution')
ax2.set_xlabel('X (m)', fontsize = 10)
ax2.set_ylabel('Y (m)', fontsize = 10)
ax2.set_zlabel('Elevation (m)', fontsize = 10)
ax2.plot_surface(X, Y, Z, cmap = 'jet')

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