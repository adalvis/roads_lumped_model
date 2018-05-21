"""
Purpose: Basic stochastic truck pass erosion model
Update: Added deposition, ditchline (03/27/2018)
Update: Create field for roughness values (04/23/2018)
Update: Water depth plots for different time slices (05/06/2018)
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
from landlab.components import LinearDiffuser, KinwaveImplicitOverlandFlowADM
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

mg = RasterModelGrid(355,51,0.225) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
init_elev = np.zeros(mg.number_of_nodes, dtype = float) #initialize the elevation grid
z = mg.add_field('topographic__elevation', init_elev, at = 'node') #create the topographic__elevation field
z_active = mg.add_zeros('node','active__elevation')


mg.set_closed_boundaries_at_grid_edges(True, True, True, True) 

road_peak = 20 #peak crowning height occurs at this x-location
up = 0.0067 #rise of slope from ditchline to crown
down = 0.0035 #rise of slope from crown to hillslope

for g in range(0,355): #loop through road length
    elev = 0 #initialize elevation placeholder
    
    for h in range(0, 51): #loop through road width
        z[g*51 + h] = elev #update elevation based on x & y locations
        z_active[g*51+h] = elev
        
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

z += mg.node_y*0.05 #add longitudinal slope to road segment
z_active +=mg.node_y*0.05
#%% Create raster of Manning's n values
init_rough = np.zeros(mg.number_of_nodes, dtype = float) #initialize the roughness grid 
n_erode = mg.add_field('roughness', init_rough, at = 'node') #create roughness field

roughness = 0.1 #initialize roughness placeholder            

for g in range(0,355): #loop through road length
    for h in range(0, 51): #loop through road width
        n_erode[g*51 + h] = roughness #update roughness values based on x & y locations
        
        if h >= 0 and h <= 4: #ditchline Manning's n value is higher than OF
            roughness = 0.1
        else:
            roughness = 0.02 

#%% Time to try a basic model!

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

#initialize truck pass and time arrays
truck_pass = []
time = []

#define how long to run the model
model_end = 10 #days

#initialize LinearDiffuser component
lin_diffuse = LinearDiffuser(mg, linear_diffusivity = 0.0001)

for i in range(0, model_end): #loop through model days
    #initialize/reset the times for each loop
    t_recover = 0
    t_pass = 0
    t_total = 0

    while t_total <=24:
        if t_total < 4:
            T_B_morning = rnd.expovariate(1/4)
            time.append(t_total+24*i)
            truck_pass.append(0)
            t_recover += T_B_morning
        elif t_total >= 4 and t_total <= 15:
            t_b = rnd.expovariate(1/2.2)
            z_active[tire_track_1] -= 0.001
            z_active[tire_track_2] -= 0.001
            z_active[out_tire_1] += 0.0004
            z_active[out_tire_2] += 0.0004
            z_active[back_tire_1] += 0.0002
            z_active[back_tire_2] += 0.0002
            time.append(t_total+24*i)
            truck_pass.append(1)
            t_pass += t_b              
        elif t_total > 15:
            T_B_night = rnd.expovariate(1/9)
            lin_diffuse.run_one_step(T_B_night)
            time.append(t_total+24*i)
            truck_pass.append(0)
            t_recover += T_B_night                
        
        t_total = t_pass + t_recover
   

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
plt.figure(figsize = (4,10))
imshow_grid(mg, z_active-z, var_name = 'Active Layer', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_0.05_rills.png', bbox_inches = 'tight')
plt.show()

#%% Plot 3D surface with rills
X_erode = mg.node_x.reshape(mg.shape)
Y_erode = mg.node_y.reshape(mg.shape)
z = z.reshape(mg.shape)

fig = plt.figure()
ax_erode = fig.add_subplot(111, projection='3d')
ax_erode.get_proj = lambda: np.dot(Axes3D.get_proj(ax_erode), np.diag([1, 3, 0.5, 1]))
ax_erode.plot_surface(X_erode, Y_erode, z)
ax_erode.view_init(elev=15, azim=-105)

ax_erode.set_xlim(0, 11)
ax_erode.set_ylim(0, 80)
ax_erode.set_zlim(0, 4)
ax_erode.set_zticks(np.arange(0, 5, 1))
ax_erode.set_xlabel('Road Width (m)')
ax_erode.set_ylabel('Road Length (m)')
ax_erode.set_zlabel('Elevation (m)')
plt.title('Road Surface Elevation', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RoadSurface_3D_0.05_rills.png')
plt.show()

#%% Add KinwaveImplicitOverlandFlow

outlet_id_1 = mg.core_nodes[np.argmin(mg.at_node['topographic__elevation'][mg.core_nodes])]                     
outlet_id_2 = tire_track_1[1]
outlet_id_3 = tire_track_2[1]
outlet_id_4 = 100
outlet_id_5 = outlet_id_2 + 2
outlet_id_6 = outlet_id_3 - 2

mg.set_watershed_boundary_condition_outlet_id(outlet_id_1, z)
mg.set_watershed_boundary_condition_outlet_id(outlet_id_2, z)
mg.set_watershed_boundary_condition_outlet_id(outlet_id_3, z)
mg.set_watershed_boundary_condition_outlet_id(outlet_id_4, z)
mg.set_watershed_boundary_condition_outlet_id(outlet_id_5, z)
mg.set_watershed_boundary_condition_outlet_id(outlet_id_6, z)

#%%

elapsed_time = 0
model_run_time = 7200
storm_duration = 3600
rr = 20

#knwv = KinwaveImplicitOverlandFlow(mg, roughness = 0.02, runoff_rate = rr, depth_exp = 1.6666667)
knwv = KinwaveImplicitOverlandFlowADM(mg, runoff_rate = rr, depth_exp = 1.6666667)

hydrograph_time = []
discharge_at_outlet_1 = []
discharge_at_outlet_2 = []
discharge_at_outlet_3 = []
discharge_at_outlet_4 = []
discharge_at_outlet_5 = []
discharge_at_outlet_6 = []
dt = 100
vol = 0


#run the model
while elapsed_time <= model_run_time:
    if elapsed_time < storm_duration:
        knwv.run_one_step(dt, current_time = elapsed_time)
    else:
        knwv.run_one_step(dt, current_time = elapsed_time, runoff_rate = 0.0)

    q_at_outlet_1 = mg.at_node['surface_water_inflow__discharge'][outlet_id_1]
    q_at_outlet_2 = mg.at_node['surface_water_inflow__discharge'][outlet_id_2]
    q_at_outlet_3 = mg.at_node['surface_water_inflow__discharge'][outlet_id_3]
    q_at_outlet_4 = mg.at_node['surface_water_inflow__discharge'][outlet_id_4]
    q_at_outlet_5 = mg.at_node['surface_water_inflow__discharge'][outlet_id_5]
    q_at_outlet_6 = mg.at_node['surface_water_inflow__discharge'][outlet_id_6]

    hydrograph_time.append(elapsed_time/3600.)
    discharge_at_outlet_1.append(q_at_outlet_1)
    discharge_at_outlet_2.append(q_at_outlet_2)
    discharge_at_outlet_3.append(q_at_outlet_3)
    discharge_at_outlet_4.append(q_at_outlet_4)
    discharge_at_outlet_5.append(q_at_outlet_5)
    discharge_at_outlet_6.append(q_at_outlet_6)
    
    time = elapsed_time/3600.
    
    if elapsed_time == 200 or elapsed_time == 400 or elapsed_time == 1800 or elapsed_time == 3600 or elapsed_time == 4100:
        plt.figure(figsize = (4,10))
        imshow_grid(mg, 'surface_water__depth', var_name = 'Water depth', 
                    var_units = 'm', grid_units = ('m','m'), cmap = 'jet', limits = (0, 0.04))
        plt.title('Water depth at time = %0.2f hr' % time, fontweight = 'bold')
#        plt.savefig('C:/Users/Amanda/Desktop/WaterDepth_rasterN%0.2f.png' % time)
        plt.show()  
    
    vol = vol + dt*(q_at_outlet_1 + q_at_outlet_2 + q_at_outlet_3 + q_at_outlet_4
                 + q_at_outlet_5 + q_at_outlet_6)
                      
    elapsed_time += dt
    

#plot the hydrograph
fig = plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.minorticks_on()
plt.plot(hydrograph_time, discharge_at_outlet_1, '-', color = '#33FFF0', label = 'Ditchline outlet')
plt.plot(hydrograph_time, discharge_at_outlet_2, '-', color = '#FF3333', label = 'Tire track 1')
plt.plot(hydrograph_time, discharge_at_outlet_3, '-', color = '#4CFF33', label = 'Tire track 2')
plt.plot(hydrograph_time, discharge_at_outlet_4, '-', color = '#C679FF', label = 'Road outlet')
plt.plot(hydrograph_time, discharge_at_outlet_5, '-', color = '#E9FF33', label = 'Inside tire track 1')
plt.plot(hydrograph_time, discharge_at_outlet_6, '-', color = '#FF339F', label = 'Inside tire track 2')
plt.legend()
plt.xlabel('Time (hr)', fontweight = 'bold')
plt.ylabel('Discharge (cms)', fontweight = 'bold')
plt.title('Outlet Hydrograph', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/OutletHydrograph_rasterN_2.png')
plt.show()