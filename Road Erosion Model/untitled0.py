"""
Created on Sat Jun  2 14:35:26 2018

Author: Amanda
"""

from landlab.io import read_esri_ascii
from landlab.plot import imshow_grid
import matplotlib.pyplot as plt


#%% RW
plt.figure()
RW_before = 'RW_before.txt'
(mg_RW_before, z) = read_esri_ascii(RW_before, name = 'topographic__elevation')

mg_RW_before.set_nodata_nodes_to_closed(z, nodata_value=-9999)

imshow_grid(mg_RW_before, 'topographic__elevation', plot_name = 'Rock Wall - before correction', 
            var_name = 'Change in elevation', var_units = 'm',grid_units = ('m','m'), 
            cmap = 'RdBu', limits = (-0.03, 0.03))
plt.show()


plt.figure()
RW_after = 'RW_after.txt'
(mg_RW_after, z) = read_esri_ascii(RW_after, name = 'topographic__elevation')

mg_RW_after.set_nodata_nodes_to_closed(z, nodata_value=-9999)

imshow_grid(mg_RW_after, 'topographic__elevation', plot_name = 'Rock Wall - after correction', 
            var_name = 'Change in elevation', var_units = 'm',grid_units = ('m','m'), 
            cmap = 'RdBu', limits = (-0.03, 0.03))
plt.show()

#%% CUH


plt.figure()
CUH_before = 'CUH_before.txt'
(mg_CUH_before, z) = read_esri_ascii(CUH_before, name = 'topographic__elevation')

mg_CUH_before.set_nodata_nodes_to_closed(z, nodata_value=-9999)

imshow_grid(mg_CUH_before, 'topographic__elevation', plot_name = 'Center for Urban Horticulture - before correction', 
            var_name = 'Change in elevation', var_units = 'm',grid_units = ('m','m'), 
            cmap = 'RdBu', limits = (-0.03, 0.03))
plt.show()


plt.figure()
CUH_after = 'CUH_after.txt'
(mg_CUH_after, z) = read_esri_ascii(CUH_after, name = 'topographic__elevation')

mg_CUH_after.set_nodata_nodes_to_closed(z, nodata_value=-9999)

imshow_grid(mg_CUH_after, 'topographic__elevation', plot_name = 'Center for Urban Horticulture - after correction', 
            var_name = 'Change in elevation', var_units = 'm',grid_units = ('m','m'), 
            cmap = 'RdBu', limits = (-0.03, 0.03))
plt.show()