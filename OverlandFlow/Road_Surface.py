"""
Purpose: Road segment grid creation based on figure conceptualizations
Date: 02/09/2018
Author: Amanda Manaster
"""

from landlab import RasterModelGrid
import matplotlib.pyplot as plt
import numpy as np
from landlab.plot.imshow import imshow_grid

grid = RasterModelGrid(355,47,0.225)
elev = np.zeros(grid.number_of_nodes, dtype = float)

max_elev = 0.109728 #meters
diff = 0.225

for i in range(0,30):
    elev[i:16685:47] = max_elev
    elev[47-i:16685:47] = max_elev
    max_elev += diff
    
imshow_grid(grid, elev)
