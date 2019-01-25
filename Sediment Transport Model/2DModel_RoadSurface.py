"""
Author: Amanda Manaster
Date: 2019-01-22
Purpose: Run 2DModel_MassBalance_OrigEqn on road surface.
"""

#%% Load python packages and set some defaults

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid 
from landlab.components import FlowAccumulator
from landlab.plot.imshow import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#for keeping track of how long the model runs
from datetime import datetime
start_time = datetime.now()

#%% Where are the truck tires on the elevation map?

# From centerline (road_peak), the truck will extend 3 cells on either side. The tires 
# themselves are the 4th cell from road_peak. This model assumes a perfect world in 
# which the truck drives symmetrically about the road's crown. For this model, I assumed that
# the truck is 1.8m wide, with the tires being 1.35m apart.

#tire_1 = 16 #x-position of one tire
#tire_2 = 24 #x-position of other tire
#
#out_1 = [15,17] #x-positions of the size cells of the first tire
#out_2 = [23,25] #x-positions of the size cells of the other tire
#
#back_tire_1 = [] #initialize the back of tire recovery for first tire
#back_tire_2 = [] #initialize the back of tire recovery for other tire

#%% Create erodible grid
def ErodibleGrid(nrows,ncols,spacing):    
    mg = RasterModelGrid(nrows,ncols,spacing) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
    z = mg.add_zeros('node','topographic__elevation') #create the topographic__elevation field
    z_sediment = mg.add_zeros('node','soil__depth')
    D_rd = mg.add_zeros('node','rainfall_detachment')
    
    
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False) 
    
    road_peak = 20 #peak crowning height occurs at this x-location
    up = 0.0067 #rise of slope from ditchline to crown
    down = 0.0035 #rise of slope from crown to hillslope
    
    for g in range(nrows): #loop through road length
        elev = 0 #initialize elevation placeholder
        
        for h in range(ncols): #loop through road width
            z[g*ncols + h] = elev #update elevation based on x & y locations
            
            if h == 0 or h == 4:
                elev = 0
            elif h == 1 or h == 3:
                elev = -0.5715
            elif h == 2:
                elev = -0.762
            elif h < road_peak and h > 3: #update latitudinal slopes based on location related to road_peak
                elev += up
            else:
                elev -= down
    
    z += mg.node_y*0.05 #add longitudinal slope to road segmen


    n = mg.add_zeros('node','roughness') #create roughness field
    
    roughness = 0.1 #initialize roughness placeholder            
    
    for g in range(nrows): #loop through road length
        for h in range(ncols): #loop through road width
            n[g*ncols + h] = roughness #update roughness values based on x & y locations
            
            if h >= 0 and h <= 4: #ditchline Manning's n value is higher than OF
                roughness = 0.1
            else:
                roughness = 0.02
                
    return(mg, z, z_sediment, D_rd, n)           

#%% Time to try a basic model!

mg, z, z_sediment, D_rd, n = ErodibleGrid(355,51,0.225)


X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)


qs_in = np.zeros(mg.number_of_nodes)
qs_out = np.zeros(mg.number_of_nodes)
dqs_dx = np.zeros(mg.number_of_nodes)

dzdt = np.zeros(mg.number_of_nodes)
dt = 0.01
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
        
        ordered_nodes = np.flipud(mg.at_node['flow__upstream_node_order'])
        dzdx = mg.calc_slope_at_node(z)
        
        flow_receiver = mg.at_node['flow__receiver_node']
        
        if t == 0:
            qs_out = k_t * da**m * 0.05**n
        else:
            qs_out = k_t * da**m * dzdx**n
        
        for i in range(mg.number_of_nodes):
            node = ordered_nodes[i]
            
            if mg.cell_area_at_node[node] == 0:
                dqs_dx[node] = 0
            else:
                dqs_dx[node] = (qs_out[node] - qs_in[node])/ mg.cell_area_at_node[node]
    
        qs_in[flow_receiver[node]] = qs_out[node]
    
        z0 = z.copy()
        z = z0 - dqs_dx*dt
    
        
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