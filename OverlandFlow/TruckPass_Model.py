"""
Purpose: Basic stochastic truck pass erosion model - no deposition
Date: 03/12/2018
Author: Amanda Manaster
"""
#%% Load python packages and set some defaults

import numpy as np
import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'
#%% Where are the truck tires on the elevation map?

# From centerline (road_peak), the truck will extend 3 cells on either side. The tires 
# themselves are the 4th cell from road_peak. This model assumes a perfect world in 
# which the truck drives symmetrically about the road's crown. For this model, I assumed that
# the truck is 1.8m wide, with the tires being 1.35m apart.

tire_1 = 12 #x-position of one tire
tire_2 = 20 #x-position of other tire

#%% Create erodible grid

mg_erode = RasterModelGrid(355,47,0.225) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
init_elev = np.zeros(mg_erode.number_of_nodes, dtype = float) #initialize the elevation grid
z_erode = mg_erode.add_field('topographic__elevation', init_elev, at = 'node') #create the topographic__elevation field

road_peak = 16 #peak crowning height occurs at this x-location
up = 0.0067 #rise of slope from ditchline to crown
down = 0.0035 #rise of slope from crown to hillslope

for g in range(0,355): #loop through road length
    elev = 0 #initialize elevation placeholder
    
    for h in range(0, 47): #loop through road width
        z_erode[g*47 + h] = elev #update elevation based on x & y locations
        
        if h < road_peak: #update latitudinal slopes based on location related to road_peak
            elev += up
        else:
            elev -= down

z_erode += mg_erode.node_y*0.01 #add longitudinal slope to road segment

#%% Time to try a basic model!

tire_track_1 = mg_erode.nodes[:, tire_1]
tire_track_2 = mg_erode.nodes[:, tire_2]

truck_pass = []
time = []

t = 0
while t <= 9:
    T_x = rnd.expovariate(1/1.8)
    z_erode[tire_track_1] -= 0.001
    z_erode[tire_track_2] -= 0.001
    time.append(t)
    truck_pass.append(1)
    t += T_x
    
#%% Plot truck passes
#x_axis = np.linspace(0,10,10)
#a = [0,1]
    
#plt.figure(figsize = (10, 4))
#plt.bar(time, truck_pass, color = 'r', edgecolor = 'k')
#plt.xticks(x_axis, np.linspace(0,10,10, dtype = int))
#plt.yticks(a, ('No','Yes'))
#plt.xlim(0,10)
#plt.ylim(0,1.1)
#plt.xlabel('Time (Days)')
#plt.ylabel('Truck Pass?')
##plt.savefig('C://Users/Amanda/Desktop/TruckPass_YN.png', bbox_inches = 'tight')
#plt.show()

#%% Plot 2D surface with rills        
plt.figure(figsize = (4,10))
imshow_grid(mg_erode, z_erode, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_0.01_rills.png', bbox_inches = 'tight')
plt.show()

#%% Plot 3D surface with rills
X_erode = mg_erode.node_x.reshape(mg_erode.shape)
Y_erode = mg_erode.node_y.reshape(mg_erode.shape)
Z_erode = z_erode.reshape(mg_erode.shape)

fig = plt.figure()
ax_erode = fig.add_subplot(111, projection='3d')
ax_erode.get_proj = lambda: np.dot(Axes3D.get_proj(ax_erode), np.diag([1, 3, 0.5, 1]))
ax_erode.plot_surface(X_erode, Y_erode, Z_erode, cmap = 'gist_earth')
ax_erode.view_init(elev=15, azim=-105)

ax_erode.set_xlim(0, 11)
ax_erode.set_ylim(0, 80)
ax_erode.set_zlim(0,2)
ax_erode.set_zticks(np.arange(0, 2.5, 0.5))
ax_erode.set_xlabel('Road Width (m)')
ax_erode.set_ylabel('Road Length (m)')
ax_erode.set_zlabel('Elevation (m)')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_3D_0.01_rills.png')
plt.show()