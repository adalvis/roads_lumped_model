"""
Author: Amanda Manaster
Date: 04/14/2017
Purpose: Define the function to create water depth plots
"""
def water_depth_plots(mg, z, t=7200, inc=72, elapsed_time=0.0, rainfall_mmhr=10, 
                      storm_duration=3600, n=0.03):
    
    #import usual Python packages
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    #import necessary Landlab components
    from landlab.components import OverlandFlow

    #import Landlab utilities
    from landlab.plot import imshow_grid
    
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.stretch'] = 1
    mpl.rcParams['font.weight'] = 'medium'
    mpl.rcParams['axes.labelweight'] = 'bold'
    
    #determine the outlet node based on the lowest elevation and set BC + links
    #    connecting to the outlet
    outlet_id = mg.core_nodes[np.argmin(mg.at_node['topographic__elevation'][mg.core_nodes])]                     
    mg.set_watershed_boundary_condition_outlet_id(outlet_id, z)
    
    of = OverlandFlow(mg, mannings_n = n, steep_slopes = True)
    
    x = np.linspace(0, t, inc)

    for i in range(72):
        model_run_time_new = x[i]
        hydrograph_time = []
        discharge_at_outlet = []

        while elapsed_time <= model_run_time_new:
            of.dt = of.calc_time_step()
            if elapsed_time < (storm_duration):
                of.rainfall_intensity = rainfall_mmhr * (2.777778*10**-7)
            else:
                of.rainfall_intensity = 0.0
            of.overland_flow()
    
            mg.at_node['surface_water__discharge'] = of.discharge_mapper(of.q, 
                  convert_to_volume = True)
            q_at_outlet = mg.at_node['surface_water__discharge'][outlet_id]

            hydrograph_time.append(elapsed_time/3600.)
            discharge_at_outlet.append(q_at_outlet)

            q_at_node = of.discharge_mapper(of.q, convert_to_volume = False)
            Fr = (q_at_node/of.h)/np.sqrt(of.g*of.h)
            Fr[np.where(Fr > 2)] = 2
            
            elapsed_time += of.dt
         
    
        time = model_run_time_new/3600.
        
        #imshow_grid(mg, Fr, plot_name = 'Froude number at time =%0.2f hr' % time, 
                     #var_name = 'Froude number', var_units = '-', 
                     #grid_units = ('m','m'), cmap = 'jet')
        #plt.show()
        
        imshow_grid(mg, 'surface_water__depth', var_name = 'Water depth', 
                    var_units = 'm', grid_units = ('m','m'), cmap = 'jet', 
                    limits = (0, 0.05))
        plt.title('Water depth at time = %0.2f hr' % time, fontweight = 'bold')
        #plt.savefig('C:/Users/Amanda/Desktop/Output/10mmph/WaterDepth%i.png' % i)
        plt.show()  
        
    return(Fr)