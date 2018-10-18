"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Test defined functions and calculate values @ t = 1
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

#from landlab.plot.imshow import imshow_grid
from landlab.plot.drainage_plot import drainage_plot
from landlab.components import FastscapeEroder, DepressionFinderAndRouter

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#%% Create initial drainage network using FastScapeEroder


fr = FlowAccumulator(mg, flow_director='D8')
fsc = FastscapeEroder(mg, K_sp=.01, m_sp=.5, n_sp=1)
df = DepressionFinderAndRouter(mg)

fsc_dt = 100.

for x in range(100):
    fr.run_one_step()
    df.map_depressions()
    flooded = np.where(df.flood_status == 3)[0]
    fsc.run_one_step(dt=fsc_dt, flooded_nodes=flooded)
    mg.at_node['topographic__elevation'][0] -= 0.001 # Uplift



#%% Plot drainage area and flow accumulation initially

ordered_nodes, mg, da = ordered(mg)

plt.figure()
drainage_plot(mg, 'drainage_area')

#%% Initial conditions slope-area plot

dzdx, mg = calculate_slope(mg, z)

plt.figure()
plt.title('Slope-Area Plot for Initial Conditions')
plt.xlabel('Area (m$^2$)')
plt.ylabel('Slope (-)')
plt.loglog(mg.at_node['drainage_area'], dzdx, 'ko')
plt.xlim(10**(0)-10, 10**(4) + 5000)   
plt.ylim(10**(-4), 10**(-1)) 
plt.show()


#%% Calculate initial sediment discharge and plot

qs, mg = sed_disch(qs, mg, ordered_nodes, da, dzdx)

plt.figure()
imshow_grid(mg, qs, plot_name = 'Sediment Discharge at t = 0', var_name = 'Sediment Discharge', 
            var_units = r'$\frac{Q^3}{s}$', grid_units = ('m','m'), cmap = 'jet') 


#%% Calculate initial sediment divergence and plot

dqs_dx = div_qs(qs, mg)

plt.figure()
imshow_grid(mg, dqs_dx, plot_name = 'Divergence of Sediment Discharge at t = 0', var_name = 'Sediment Flux', 
            var_units = r'$\frac{Q^2}{s}$', grid_units = ('m','m'), cmap = 'jet')

#%% Solve for elevation change for 1 time step

z, dzdt = ExnerSolver(mg, z, dzdt, ordered_nodes, dqs_dx, 1)

plt.figure()
imshow_grid(mg, z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
           var_units = 'm', grid_units = ('m','m'), cmap = 'jet')

#%% Solve for Q_out

qs = Q_out(qs, dzdt, ordered_nodes, mg)

plt.figure()
imshow_grid(mg, qs, plot_name = 'Sediment discharge moving out', var_name = 'Sediment Discharge',
            var_units = r'$\frac{Q^3}{s}$', grid_units = ('m','m'), cmap = 'jet')






