"""
Purpose: Add rainfall detachment to road erosion model
Date: 05/16/2018
Author: Amanda
"""

#%% Load python packages and set some defaults

import numpy as np
import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid 
from landlab.components import LinearDiffuser
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#%% Rainfall Detachment

# Use ***modified*** equation from Nord & Esteves (2005) for rainfall detachment:

#   D_{rd_d} = \alpha R^{p} (1 - \frac{h}{z_{m}}) [kg m^-2 s^-1]
#       where D_{rd_d} = rainfall detachment (original soil)
#             \alpha = rainfall erodibility of original soil [kg m^-2 mm^-1]
#             R = rainfall intensity [m s^-1]
#             p = empirical parameter, set to 1.0
#             h = flow depth [m]
#             z_{m} = 3 * (2.23 * R^{0.182})