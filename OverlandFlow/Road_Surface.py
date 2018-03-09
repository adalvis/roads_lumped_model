"""
Purpose: Road segment grid creation based on figure conceptualizations
Date: 02/09/2018
Author: Amanda Manaster
"""

from landlab import RasterModelGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

mg = RasterModelGrid(355,47,0.225) #produces an 80m x 10.67m
elev = np.zeros(mg.number_of_nodes, dtype = float) #initialize the 
z = mg.add_field('topographic__elevation', elev, at = 'node')

road_peak = 16 #peak crowning height occurs at this point
up = 0.0067 #slope from ditchline to crown
down = 0.0035 #slope from crown to hillslope

for i in range(0,355):
    init_elev = 0
    
    for j in range(0, 47):
        z[i*47 + j] = init_elev
        
        if j < road_peak:
            init_elev += up
        else:
            init_elev -= down

#z = z + mg.node_y*0.001
    
plt.figure(figsize = (4,10))
imshow_grid(mg, z, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth'
            )
plt.title('Road Surface Elevation', fontweight = 'bold')
plt.savefig('C://Users/Amanda/Desktop/RoadSurface.png', bbox_inches = 'tight')
plt.show()

# =============================================================================
# Original code from Sai
# =============================================================================

#from landlab import RasterModelGrid
#import numpy as np
#from landlab.io import read_esri_ascii
#from landlab.plot.imshow import imshow_field, imshow_grid

#grid = RasterModelGrid(53,67,10.)
#elev = np.zeros(grid.number_of_nodes, dtype = float)
#
### For reference - find factors of a number
##def factors(n):
##    return set(reduce(list.__add__, ([i, n//i] for i in range(1, \
#                #int(n**0.5) + 1) if n % i == 0)))
#
#max_elev = 1600.
#diff = 10.
#for i in range(0,30):
#    elev[i:3551:67] = max_elev
#    elev[67-i:3551:67] = max_elev
#    max_elev -= diff
#
#for i in range(30,38):
#    elev[i:3551:67] = max_elev