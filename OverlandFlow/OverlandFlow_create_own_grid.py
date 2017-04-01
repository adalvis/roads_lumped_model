"""
Author: Amanda Manaster
Date: 04/01/2017
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
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05, at = 'node')

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
imshow_grid(mg, 'topographic__elevation', plot_name = 'Topographic Elevation', 
            var_name = 'Elevation', var_units = 'm',grid_units = ('m','m'), 
            cmap = 'gist_earth')
plt.savefig(r'C:\Users\Amanda\Desktop\Elevation.png')
plt.show()

#initialize variables to run the model
elapsed_time = 0.0
model_run_time = 5400. #1.5 hr
storm_duration = 1800.0 #storm lasts 0.5 hr
rainfall_mmhr = 5.
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
plt.plot(hydrograph_time, discharge_at_outlet, 'k-')
axes = plt.gca()
plt.xlabel('Time (hr)', fontweight = 'bold')
plt.ylabel('Discharge (cms)', fontweight = 'bold')
plt.title('Outlet Hydrograph, Rainfall: 5 mm/hr in 0.5 hr', fontweight = 'bold')
plt.savefig(r'C:\Users\Amanda\Desktop\Hydrograph.png')
plt.show()

#re-initialize variables to get water depth map at t = 1hr
elapsed_time = 0.0
hydrograph_time = []
discharge_at_outlet = []

while elapsed_time < storm_duration:
    of.dt = of.calc_time_step()
    of.rainfall_intensity = rainfall_mmhr * (2.777778*10**-7)
    of.overland_flow()
    
    mg.at_node['surface_water__discharge'] = (map_max_of_inlinks_to_node(mg, 
               np.abs(of.q)*mg.dx))

    hydrograph_time.append(elapsed_time/3600.)
    discharge_at_outlet.append(np.abs(of.q[outlet_link])*mg.dx)
                                           
    elapsed_time += of.dt

imshow_grid(mg, 'surface_water__depth', plot_name 
            = 'Water depth at time = 0.5 hr', var_name = 'Water depth', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'Blues')
plt.savefig(r'C:\Users\Amanda\Desktop\WaterDepth.png')
plt.show()


