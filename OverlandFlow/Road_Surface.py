"""
Purpose: Road segment grid creation based on figure conceptualizations
Date: 03/09/2018
Author: Amanda Manaster
"""
#%% Load python packages and set some defaults

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#%% 2D surface elevation plot of road surface

mg = RasterModelGrid(355,47,0.225) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
init_elev = np.zeros(mg.number_of_nodes, dtype = float) #initialize the elevation grid
z = mg.add_field('topographic__elevation', init_elev, at = 'node') #create the topographic__elevation field

road_peak = 16 #peak crowning height occurs at this x-location
up = 0.0067 #rise of slope from ditchline to crown
down = 0.0035 #rise of slope from crown to hillslope

for i in range(0,355): #loop through road length
    elev = 0 #initialize elevation placeholder
    
    for j in range(0, 47): #loop through road width
        z[i*47 + j] = elev #update elevation based on x & y locations
        
        if j < road_peak: #update latitudinal slopes based on location related to road_peak
            elev += up
        else:
            elev -= down

z = z + mg.node_y*0.01 #add longitudinal slope to road segment
    
plt.figure(figsize = (4,10))
imshow_grid(mg, z, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_0.01.png', bbox_inches = 'tight')
plt.show()

#%% 3D plot of road surface

X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

fig = plt.figure(figsize = (4,10)) #how to update 3d plot
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap = 'gist_earth')
ax.view_init(elev=35, azim=-120)

ax.set_xlim(0, 11)
ax.set_ylim(0, 80)
ax.set_zlim(0,2)
ax.set_xlabel('Road Width (m)')
ax.set_ylabel('Road Length (m)')
ax.set_zlabel('Elevation (m)')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_3D_0.01.png', bbox_inches = 'tight')
plt.show()



#%% Make same surface plot as above showing where the tire tracks will form

mg_1 = RasterModelGrid(355,47,0.225) #produces an 80m x 10.67m
init_elev = np.zeros(mg.number_of_nodes, dtype = float) #initialize the elevation grid
z_1 = mg_1.add_field('topographic__elevation', init_elev, at = 'node')

road_peak = 16 #peak crowning height occurs at this x-location
up = 0.0067 #rise of slope from ditchline to crown
down = 0.0035 #rise of slope from crown to hillslope

for i in range(0,355): #loop through road length
    elev = 0 #initialize elevation
    
    for j in range(0, 47): #loop through road width
        z_1[i*47 + j] = elev #update elevation based on location
        
        if j < road_peak: #update latitudinal slopes based on location related to road_peak
            elev += up
        else:
            elev -= down

z_1 = z_1 + mg_1.node_y*0.01 #add longitudinal slope to road segment
    
plt.figure(figsize = (4,10))
imshow_grid(mg_1, z_1, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.plot((2.925, 2.925), (0,79.875), 'r-')
plt.plot((4.275, 4.275), (0,79.875), 'r-')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_truck_0.01.png', bbox_inches = 'tight')
plt.show()

#%% Where are the truck tires on the elevation map?

# From centerline (road_peak), the truck will extend 3 cells on either side. The tires 
# themselves are the 4th cell from road_peak. This model assumes a perfect world in 
# which the truck drives symmetrically about the road's crown.

tire_1 = 12 #x-position of one tire
tire_2 = 20 #x-position of other tire

#%% Time to try a basic model!


for i in range(0,355): #loop through road length
    for j in range(0, 47): #loop through road width
        if j == tire_1 or j == tire_2: #update latitudinal slopes based on location related to road_peak
            z[i*47 + j] -= 0.5
            
plt.figure(figsize = (4,10))
imshow_grid(mg, z, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_0.01.png', bbox_inches = 'tight')
plt.show()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            