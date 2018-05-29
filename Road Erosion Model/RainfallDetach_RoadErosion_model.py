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

def RainfallDetachment(mg, R, p = 1.0, alpha = 0.35):
    
    z_m = 3 * (2.23 * R^{0.182})
    
    D_rd = alpha*R^p * (1-mg.at_node('surface_water__depth')/z_m)
    
    return(D_rd)