"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Run model until equilibrium is reached
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

#from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#%% Create model driver

U = 5e-5
Ua = U * da
Ua[0] = 0
t = 0
dt = 1


#%%

while t <= 100:

    ordered_nodes, mg, da = ordered(mg, fa)
    dzdx = calculate_slope(mg, z)
    
    qs_in[:]=0
    qs, qs_in, mg = sed_disch(qs, qs_in, mg, z, ordered_nodes, da, dzdx, dzdt)
    dqs_dx, qs_link = div_qs(qs, z, mg)
    z, dzdt = ExnerSolver(mg, z, dzdt, ordered_nodes, dqs_dx, dt)
    
    t += dt
    print(t)
      
    
plt.figure()
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

print(np.sum(z-z_init))
