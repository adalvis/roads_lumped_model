"""
Author: Amanda Manaster
Date: 04/26/2019
Purpose: Create two different domains for preliminary model for sediment trap efficiency experiment.
"""
#%% Load python packages and set some defaults

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid
from landlab.components import LinearDiffuser
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#%% 2D surface elevation plot of road surface

mgRoad = RasterModelGrid(355,47,0.225) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)

np.random.seed(2)
surfaceRoad = np.random.rand(mgRoad.number_of_nodes)/55 #initialize surface

zRoad = mgRoad.add_zeros('topographic__elevation', at = 'node') #create topographic__elevation field

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
            
        zRoad[i*47 + j] = elev #update elevation based on x & y locations    

zRoad += mgRoad.node_y*0.05 + surfaceRoad #add longitudinal slope to road segment
    
plt.figure(figsize = (4,10))
imshow_grid(mgRoad, zRoad, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'terrain')
plt.title('Road Surface Elevation', fontweight = 'bold')

#%% 2D surface elevation plot of ditchline

mgDitch = RasterModelGrid(800, 6, 0.1)
zDitch = mgDitch.add_zeros('topographic__elevation', at = 'node')

np.random.seed(2)
surfaceDitch = np.random.rand(mgDitch.number_of_nodes)/55

zDitch += surfaceDitch + mgDitch.node_y*0.05

mgDitch.set_fixed_value_boundaries_at_grid_edges(True, False, True, False)
mgDitch.set_closed_boundaries_at_grid_edges(False, True, False, True)  

ld=0.8
total_t=150
dt=10
erosion_rate=-0.07

lin_diffuse = LinearDiffuser(mgDitch, linear_diffusivity = ld)
nt = int(total_t // dt)
for i in range(nt):
    lin_diffuse.run_one_step(dt)
    zDitch[mgDitch.core_nodes] += erosion_rate * dt  # erode

plt.figure()
imshow_grid(mgDitch, zDitch, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'terrain')
plt.title('Ditchline Elevation', fontweight = 'bold')