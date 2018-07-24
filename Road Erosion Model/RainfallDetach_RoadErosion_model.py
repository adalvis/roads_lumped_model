"""
Purpose: Add rainfall detachment to road erosion model
Date: 05/16/2018
Author: Amanda
"""

#%% Rainfall Detachment

# Use ***modified*** equation from Nord & Esteves (2005) for rainfall detachment:

#   D_{rd_d} = \alpha R^{p} (1 - \frac{h}{z_{m}}) [kg m^-2 s^-1]
#       where D_{rd_d} = rainfall detachment (original soil)
#             \alpha = rainfall erodibility of original soil [kg m^-2 mm^-1]
#             R = rainfall intensity [m s^-1]
#             p = empirical parameter, set to 1.0
#             h = flow depth [m]
#             z_{m} = 3 * (2.23 * R^{0.182})

def RainfallDetachment(mg, R, D_rd, p = 1.0, alpha = 0.0902):
    z_m = 3 * (2.23 * R**(0.182))
    
    for n in range(len(mg.node_x)):
        D_rd[n] = alpha*R**p * (1-mg.at_node['surface_water__depth'][n]/z_m)
    
   
    return(D_rd)
    
#d = RainfallDetachment(mg, 20, D_rd)    
#
#plt.figure(figsize = (4,10))
#imshow_grid(mg, d, var_name = 'Rainfall detachment rate', 
#            var_units = 'kg m^-2 s^-1',grid_units = ('m','m'), cmap = 'gist_earth')