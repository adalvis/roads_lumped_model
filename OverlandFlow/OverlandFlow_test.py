#Author: Amanda Manaster
#Date: 03/30/2017
#Purpose: Playing with the Landlab component OverlandFlow.

#import usual Python packages
import numpy as np
import matplotlib.pyplot as plt

#import necessary Landlab components
from landlab.components import OverlandFlow, SinkFiller

#import Landlab utilities
from landlab.io import read_esri_ascii
from landlab.plot import imshow_grid
from landlab.grid.raster_mappers import map_max_of_inlinks_to_node

#load in DEM
watershed_dem = 'Square_TestBasin.asc'
(rmg, z) = read_esri_ascii(watershed_dem, name = 'topographic__elevation')

#set boundary conditions
rmg.set_watershed_boundary_condition(z, nodata_value=-9999)

#fill sinks to ensure complete drainage
sf = SinkFiller(rmg, routing = 'D8', apply_slope = True, fill_slope = 1.e-5)
sf.fill_pits()

#look at the pretty grid
imshow_grid(rmg, 'topographic__elevation', plot_name = 'Topographic Elevation', var_name = 'Elevation', var_units = 'm',
            grid_units = ('m','m'), cmap = 'pink')
plt.show()

elapsed_time = 0.0
model_run_time = 7200.
of = OverlandFlow(rmg, steep_slopes = True)

storm_duration = 7200.0
rainfall_mmhr = 5.

outlet_link = 299
hydrograph_time = []
discharge_at_outlet = []

while elapsed_time < model_run_time:
    of.dt = of.calc_time_step()
    if elapsed_time < (storm_duration):
        of.rainfall_intensity = rainfall_mmhr * (2.777778*10**-7)
    else:
        of.rainfall_intensity = 0.0
    of.overland_flow()
    
    rmg.at_node['surface_water__discharge'] = (map_max_of_inlinks_to_node(rmg, np.abs(of.q)*rmg.dx))

    hydrograph_time.append(elapsed_time/3600.)
    discharge_at_outlet.append(np.abs(of.q[outlet_link])*rmg.dx)
                                           
    elapsed_time += of.dt
    
imshow_grid(rmg, 'surface_water__depth', plot_name = 'Water depth at time = 2 hr', var_name = 'Water depth', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'Blues')
plt.show()

plt.plot(hydrograph_time, discharge_at_outlet)
axes = plt.gca()
plt.xlabel('Time (hr)')
plt.ylabel('Discharge (cms)')
plt.title('Outlet Hydrograph, Rainfall: 5mm/hr in 2 hr')
plt.show()