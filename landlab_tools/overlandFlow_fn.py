"""
Author: Amanda Manaster
Date: 04/19/2020
Purpose: Define functions to run KinwaveImplicitOverlandFlow and OverlandFlow
         in Landlab.
"""
def kinwave_run(mg, z, s, oid, elapsed_time=100, model_run_time=7200, storm_duration=3600,
                rr = 10):
     
    #import usual Python packages
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    #import necessary Landlab components
    from landlab.components import KinwaveImplicitOverlandFlow
    
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.stretch'] = 1
    
    #determine the outlet node based on the lowest elevation and set BC + links
    #    connecting to the outlet
    outlet_id = oid
    
    knwv = KinwaveImplicitOverlandFlow(mg, runoff_rate = 0.0, roughness = 0.03, depth_exp = 1.6666667)
    
    hydrograph_time = [0]
    discharge_at_outlet = [0]
    dt = 100
    vol = 0
    
    rr = rr*2.77778E-07
    
    #run the model
    while elapsed_time <= model_run_time:
        if elapsed_time < storm_duration:
            knwv.run_one_step(dt, current_time = elapsed_time, runoff_rate = rr)
        else:
            knwv.run_one_step(dt, current_time = elapsed_time, runoff_rate = 0.0)
    
        q_at_outlet = mg.at_node['surface_water_inflow__discharge'][outlet_id]

        hydrograph_time.append(elapsed_time/3600.)
        discharge_at_outlet.append(q_at_outlet)
        
        vol += dt*q_at_outlet 
                          
        elapsed_time += dt

    #plot the hydrograph
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
                   left = 'on', top = 'on', right = 'on')
    plt.minorticks_on()
    plt.plot(hydrograph_time, discharge_at_outlet, 'k-')
    plt.ylim(0, 0.7)
    plt.xlabel('Time (hr)', fontweight = 'bold')
    plt.ylabel('Discharge (cms)', fontweight = 'bold')
    plt.title('Outlet Hydrograph, Rainfall: 200 mm/hr, Slope = %0.4f' %s, 
              fontweight = 'bold')
    plt.savefig(r'C:\Users\Amanda\Desktop\Output\200mmph\KinwaveHydrograph_%f.png' % s)
    plt.show()
    
    
    return(vol)


def of_run(mg, z, s, oid, boo=True, elapsed_time=0.0, model_run_time=7200, 
              storm_duration=3600, rainfall_mmhr=10, n=0.03):
    
    #import usual Python packages
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    #import necessary Landlab components
    from landlab.components import OverlandFlow
    
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.stretch'] = 1
    
    #determine the outlet node based on the lowest elevation and set BC + links
    #    connecting to the outlet
    outlet_id = oid
    
    of = OverlandFlow(mg, mannings_n = n, steep_slopes = boo)

    hydrograph_time = []
    discharge_at_outlet = []
    vol = 0

    #run the model
    while elapsed_time <= model_run_time:
        of.delta_t = of.calc_time_step()
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
        
        vol += of.delta_t*q_at_outlet
                                   
        elapsed_time += of.delta_t

    #plot the hydrograph
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
                   left = 'on', top = 'on', right = 'on')
    plt.minorticks_on()
    plt.plot(hydrograph_time, discharge_at_outlet, 'k-')
    plt.ylim(0, 0.35)
    plt.xlabel('Time (hr)', fontweight = 'bold')
    plt.ylabel('Discharge (cms)', fontweight = 'bold')
    plt.title('Outlet Hydrograph, Rainfall: 100 mm/hr, Slope = %0.4f' %s, 
              fontweight = 'bold')
    #plt.savefig(r'C:\Users\Amanda\Desktop\Output\200mmph\Hydrograph_%f.png' % s)
    plt.show()
    
    return(vol)