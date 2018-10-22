"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Analytical sol'n slope-area plot for comparison purposes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#%% Analytical sol'n for k_t = 0.01

def analytical_soln(grid):
    n = 2.
    m = 2.
    U = 5e-5
    k_t = 0.01
    
    slope = np.empty((10000))
    
    for node in grid.core_nodes:
        slope[node] = ((U/k_t)**(1/n)) * (grid.at_node['drainage_area'][node])**((1-m)/n)
    
    slope = slope[grid.core_nodes]
    
    return slope


ordered_nodes, mg, da, flooded = ordered(mg, fa, df, outlet_id = outlet_id)

dzdx, mg = calculate_slope(mg, z)

slope = analytical_soln(mg)
drainage_area = mg.at_node['drainage_area'][mg.core_nodes]

plt.figure()
plt.loglog(drainage_area, slope, 'b-')
plt.loglog(drainage_area, dzdx[mg.core_nodes], 'ko')
plt.title('Slope-Area Plot of Analytical Solution')
plt.xlabel('Area (m$^2$)')
plt.ylabel('Slope (-)')
#plt.xlim(0.9*10**(0), 9*10**(0))
#plt.ylim(2*10**(-2), 8*10**(-2))    
plt.show()