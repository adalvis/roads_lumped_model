"""
Purpose: Basic stochastic truck pass erosion model
Update: Added deposition, ditchline (03/27/2018)
Update: Create field for roughness values (04/23/2018)
Date: 03/12/2018
Author: Amanda Manaster
"""
#%% Load python packages and set some defaults

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid 
from landlab.components import TruckPassErosion
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=1000)

#%% Where are the truck tires on the elevation map?

# From centerline (road_peak), the truck will extend 3 cells on either side. The tires 
# themselves are the 4th cell from road_peak. This model assumes a perfect world in 
# which the truck drives symmetrically about the road's crown. For this model, I assumed that
# the truck is 1.8m wide, with the tires being 1.35m apart.

tire_1 = 16 #x-position of one tire
tire_2 = 24 #x-position of other tire

out_1 = [15,17] #x-positions of the size cells of the first tire
out_2 = [23,25] #x-positions of the size cells of the other tire

back_tire_1 = [] #initialize the back of tire recovery for first tire
back_tire_2 = [] #initialize the back of tire recovery for other tire

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

#%%
#get node IDs for the important nodes
tire_track_1 = mg.nodes[:, tire_1]
tire_track_2 = mg.nodes[:, tire_2]
out_tire_1 = mg.nodes[:, out_1]
out_tire_2 = mg.nodes[:, out_2]

back_tire_1.append(mg.nodes[0, tire_1])
back_tire_2.append(mg.nodes[0, tire_2])

for k in range(0,354):
    back_tire_1.append(mg.nodes[k+1, tire_1])
    back_tire_2.append(mg.nodes[k+1, tire_2])
    
back_tire_1_new = np.array(back_tire_1)    
back_tire_2_new = np.array(back_tire_2)


tire_tracks = np.array([tire_track_1, tire_track_2, out_tire_1[:,0], \
                        out_tire_1[:,1], out_tire_2[:,0], out_tire_2[:,1], \
                        back_tire_1_new, back_tire_2_new])

#%%
#initialize truck pass and time arrays
truck_pass = []
time = []

#define how long to run the model
model_end = 10 #days

tpe = TruckPassErosion(mg, diffusivity = 0.0001)

for i in range(0, model_end): #loop through model days
    time, truck_pass = tpe.run_one_step(tire_tracks, i)
#%%       
X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

fig = plt.figure(figsize = (5,3))
ax = fig.add_subplot(111, projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 3, 0.5, 1]))
ax.plot_surface(X, Y, Z)
ax.view_init(elev=15, azim=-105)

ax.set_xlim(0, 11)
ax.set_ylim(0, 80)
ax.set_zlim(0, 4)
ax.set_zticks(np.arange(0, 5, 1))
ax.set_xlabel('Road Width (m)', fontsize = 12)
ax.set_ylabel('Road Length (m)', fontsize = 12)
ax.set_zlabel('Elevation (m)', fontsize = 12)
plt.title('Road Surface Elevation', fontweight = 'bold',  fontsize = 20)
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_3D_0.05_rills_%i.tif' % i, dpi = 200)
plt.show()

plt.figure(figsize = (4,10))
imshow_grid(mg, z, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth', vmin = 0, vmax = 4)
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_0.05_rills_%i.tif' % i, dpi = 200, bbox_inches = 'tight')
plt.show()        
       

#%% Plot truck passes
x_axis = np.linspace(0, model_end*24, model_end+1)
a = [0,1]
    
plt.figure(figsize = (12, 5))
plt.bar(time, truck_pass, color = 'r', edgecolor = 'k')
plt.xticks(x_axis, np.linspace(0,model_end+1,model_end+2, dtype = int))
plt.yticks(a, ('No','Yes'))
plt.xlim(0,model_end*24)
plt.ylim(0,1.1)
plt.xlabel('Time (Days)')
plt.ylabel('Truck Pass?')
#plt.savefig('C://Users/Amanda/Desktop/TruckPass_YN.png', bbox_inches = 'tight')
plt.show()

#%% Plot 2D surface with rills        


##%% Plot 3D surface with rills
#X = mg.node_x.reshape(mg.shape)
#Y = mg.node_y.reshape(mg.shape)
#z = z_bed.reshape(mg.shape)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 3, 0.5, 1]))
#ax.plot_surface(X, Y, z)
#ax.view_init(elev=15, azim=-105)
#
#ax.set_xlim(0, 11)
#ax.set_ylim(0, 80)
#ax.set_zlim(0, 4)
#ax.set_zticks(np.arange(0, 5, 1))
#ax.set_xlabel('Road Width (m)')
#ax.set_ylabel('Road Length (m)')
#ax.set_zlabel('Elevation (m)')
#plt.title('Road Surface Elevation', fontweight = 'bold')
##plt.savefig('C://Users/Amanda/Desktop/RoadSurface_3D_0.05_rills.png')
#plt.show()
