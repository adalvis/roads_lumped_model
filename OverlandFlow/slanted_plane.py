"""
Created on Mon Apr 17 09:01:59 2017

Author: Amanda
"""

def slanted_plane(size=10000, height=1, s=0.05):
    #import usual Python packages
    import numpy as np
    import matplotlib.pyplot as plt

    #import Landlab utilities
    from landlab import RasterModelGrid
    from landlab.plot import imshow_grid

    #initialize a surface that has an elevation of 12m
    init = np.ones([size])
    surface = init*height

    #create a 100x100 grid and add a slanted surface elevation
    mg = RasterModelGrid(100, 100, spacing = (1,1))
    z = mg.add_field('topographic__elevation', surface + mg.node_y*s +mg.node_x*s, at = 'node')

    #set boundary conditions
    mg.set_fixed_value_boundaries_at_grid_edges(True, False, True, False)
    mg.set_closed_boundaries_at_grid_edges(False, True, False, True)                  

    #plot the grid showing its topographic elevation
    imshow_grid(mg, 'topographic__elevation', plot_name = 'Topographic Elevation, Slope = %.4f' % s, 
                var_name = 'Elevation', var_units = 'm',grid_units = ('m','m'), 
                cmap = 'gist_earth', limits= (0,10))
    #plt.savefig('C://Users/Amanda/Desktop/Output/SlantedPlane_%f.png' % s)
    plt.show()
    
    return(mg, z, s)
