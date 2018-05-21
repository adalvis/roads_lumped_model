"""
Purpose: Add KinwaveImplicitOverlandFlow to the model
Date: 05/16/2018
Author: Amanda Manaster
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from landlab.components import  KinwaveImplicitOverlandFlowADM

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'


def OverlandFlow(mg, tire_track_1, tire_track_2, z_active):
    
    outlet_id_1 = mg.core_nodes[np.argmin(mg.at_node['topographic__elevation'][mg.core_nodes])]                     
    outlet_id_2 = tire_track_1[1]
    outlet_id_3 = tire_track_2[1]
    outlet_id_4 = 100
    outlet_id_5 = outlet_id_2 + 2
    outlet_id_6 = outlet_id_3 - 2
    
    mg.set_watershed_boundary_condition_outlet_id(outlet_id_1, z_active)
    mg.set_watershed_boundary_condition_outlet_id(outlet_id_2, z_active)
    mg.set_watershed_boundary_condition_outlet_id(outlet_id_3, z_active)
    mg.set_watershed_boundary_condition_outlet_id(outlet_id_4, z_active)
    mg.set_watershed_boundary_condition_outlet_id(outlet_id_5, z_active)
    mg.set_watershed_boundary_condition_outlet_id(outlet_id_6, z_active)
    
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
        
#        time = elapsed_time/3600.
        
#        if elapsed_time == 200 or elapsed_time == 400 or elapsed_time == 1800 or elapsed_time == 3600 or elapsed_time == 4100:
#            plt.figure(figsize = (4,10))
#            imshow_grid(mg, 'surface_water__depth', var_name = 'Water depth', 
#                        var_units = 'm', grid_units = ('m','m'), cmap = 'jet', limits = (0, 0.04))
#            plt.title('Water depth at time = %0.2f hr' % time, fontweight = 'bold')
#    #        plt.savefig('C:/Users/Amanda/Desktop/WaterDepth_rasterN%0.2f.png' % time)
#            plt.show()  
        
        vol = vol + dt*(q_at_outlet_1 + q_at_outlet_2 + q_at_outlet_3 + q_at_outlet_4
                     + q_at_outlet_5 + q_at_outlet_6)
                          
        elapsed_time += dt
        
    
    #plot the hydrograph
    plt.figure()
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