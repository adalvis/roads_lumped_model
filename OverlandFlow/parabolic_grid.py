"""
Author: Amanda Manaster
Date: 04/14/2017
Purpose: Define the function to create a 100x100 parabolic grid
"""
def parabolic_grid(size=10000, height=12, s=0.05, ld=0.8, total_t=2000, dt=100, erosion_rate=-0.005):
    #import usual Python packages
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    #import necessary Landlab components
    from landlab.components import LinearDiffuser

    #import Landlab utilities
    from landlab import RasterModelGrid
    from landlab.plot import imshow_grid
    
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.stretch'] = 1
    mpl.rcParams['font.weight'] = 'medium'
    mpl.rcParams['axes.labelweight'] = 'bold'

    #initialize a surface that has an elevation of 12m
    init = np.ones([size])
    surface = init*height

    #create a 100x100 grid and add a slanted surface elevation
    mg = RasterModelGrid(100, 100, spacing = (1,1))
    z = mg.add_field('topographic__elevation', surface + mg.node_y*s, at = 'node')

    #set boundary conditions
    mg.set_fixed_value_boundaries_at_grid_edges(True, False, True, False)
    mg.set_closed_boundaries_at_grid_edges(False, True, False, True)                  

    #use LinearDiffuser to erode the core nodes; this creates a parabolic shape
    lin_diffuse = LinearDiffuser(mg, linear_diffusivity = ld)
    nt = int(total_t // dt)
    for i in range(nt):
        lin_diffuse.run_one_step(dt)
        z[mg.core_nodes] += erosion_rate * dt  # erode

    #plot the grid showing its topographic elevation
    imshow_grid(mg, 'topographic__elevation', var_name = 'Elevation', 
                var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth', 
                limits= (5.5, 17))
    plt.title('Topographic Elevation, Slope = %0.4f' %s, fontweight = 'bold')
    #plt.savefig('C://Users/Amanda/Desktop/Output/TopographicElevation_%f.png' % s)
    plt.show()
    
    return(mg, z, s)
