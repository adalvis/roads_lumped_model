"""
Author: Amanda Manaster
Date: 04/01/2017
Updated: 04/04/2017 - added for loop to get multiple water depth plots
Updated: 04/07/2017 - increased storm duration and intensity; adjusted model
                      run time; found and fixed bug that made storm intensity 
                      continue for longer than storm duration; increased number
                      of water depth plots
Updated: 04/14/2017 - adjusted model run times                      
Purpose: Playing with the Landlab component OverlandFlow. Learn to create
         grid in Landlab.
"""
#import usual Python packages
import numpy as np
import matplotlib.pyplot as plt

#import necessary Landlab components
from landlab.components import OverlandFlow, LinearDiffuser

#import Landlab utilities
from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from landlab.grid.raster_mappers import map_max_of_inlinks_to_node

#initialize a surface that has an elevation of 12m
init = np.ones([10000])
surface = init*12

#create a 100x100 grid and add a slanted surface elevation
mg = RasterModelGrid(100,100, spacing = (1,1))
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.0005, at = 'node')

#set boundary conditions
mg.set_fixed_value_boundaries_at_grid_edges(True, False, True, False)
mg.set_closed_boundaries_at_grid_edges(False, True, False, True)                  

#use LinearDiffuser to erode the core nodes; this creates a parabolic shape
lin_diffuse = LinearDiffuser(mg, linear_diffusivity=0.8)
total_t = 2000.
dt = 100.
erosion_rate = -0.005
nt = int(total_t // dt)
for i in range(nt):
    lin_diffuse.run_one_step(dt)
    z[mg.core_nodes] += erosion_rate * dt  # erode

#determine the outlet node based on the lowest elevation and set BC + links
#    connecting to the outlet
outlet_id = mg.core_nodes[np.argmin(mg.at_node['topographic__elevation'][mg.core_nodes])]                     
outlet = mg.set_watershed_boundary_condition_outlet_id(outlet_id, z)
outlet_link = mg.links_at_node[outlet_id]

#plot the grid showing its topographic elevation
plt.figure()
imshow_grid(mg, 'topographic__elevation', plot_name = 'Topographic Elevation', 
            var_name = 'Elevation', var_units = 'm',grid_units = ('m','m'), 
            cmap = 'gist_earth')
#plt.savefig(r'C:\Users\Amanda\Desktop\Elevation.png')
plt.show()

#initialize variables to run the model
elapsed_time = 0.0
model_run_time = 7200. #2 hr
storm_duration = 3600.0 #storm lasts 1 hr
rainfall_mmhr = 10.
outlet_link = outlet_link[2] #use the link that is contributing flow to outlet
hydrograph_time = []
discharge_at_outlet = []

of = OverlandFlow(mg, mannings_n = 0.03, steep_slopes = True) #set steep_slopes to True to stabilize

#run the model
while elapsed_time < model_run_time:
    of.delta_t = of.calc_time_step()
    if elapsed_time < (storm_duration):
        of.rainfall_intensity = rainfall_mmhr * (2.777778*10**-7)
    else:
        of.rainfall_intensity = 0.0
    of.overland_flow()
    
    mg.at_node['surface_water__discharge'] = (map_max_of_inlinks_to_node(mg,
               np.abs(of.q)*mg.dx))

    hydrograph_time.append(elapsed_time/3600.)
    discharge_at_outlet.append(np.abs(of.q[outlet_link])*mg.dx)
                                           
    elapsed_time += of.delta_t

#plot the hydrograph
plt.figure()
plt.plot(hydrograph_time, discharge_at_outlet, 'k-')
axes = plt.gca()
plt.xlabel('Time (hr)', fontweight = 'bold')
plt.ylabel('Discharge (cms)', fontweight = 'bold')
plt.title('Outlet Hydrograph, Rainfall: 10 mm/hr in 1 hr', fontweight = 'bold')
#plt.savefig(r'C:\Users\Amanda\Desktop\Output\10mmph\Hydrograph.png')
plt.show()

#re-initialize variables to get water depth map at different time increments
x = np.linspace(0, 7200, 10)

for i in range(10):
    elapsed_time = 0.0
    model_run_time_new = x[i]
    hydrograph_time = []
    discharge_at_outlet = []

    while elapsed_time < model_run_time_new:
        of.dt = of.calc_time_step()
        if elapsed_time < (storm_duration):
            of.rainfall_intensity = rainfall_mmhr * (2.777778*10**-7)
        else:
            of.rainfall_intensity = 0.0
        of.overland_flow()
    
        mg.at_node['surface_water__discharge'] = (map_max_of_inlinks_to_node(mg, 
                  np.abs(of.q)*mg.dx))

        hydrograph_time.append(elapsed_time/3600.)
        discharge_at_outlet.append(np.abs(of.q[outlet_link])*mg.dx)

        elapsed_time += of.dt  
    
    time = model_run_time_new/3600.
    
    plt.figure()
    imshow_grid(mg, 'surface_water__depth', plot_name 
                = 'Water depth at time = %0.2f hr' % time, 
                var_name = 'Water depth', var_units = 'm', 
                grid_units = ('m','m'), cmap = 'jet', limits= (0,0.045))
    #plt.savefig('C:/Users/Amanda/Desktop/Output/10mmph/WaterDepth%i_new.png' % i)
    plt.show()     




