"""
Author: Amanda Manaster
Date: 09/18/2018
Purpose: Trying to create Exner solver
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from landlab.components import FlowAccumulator, FlowDirectorD8
from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


#%% Create grid and plot map

init = np.ones([10000])
surface = (init*12) + np.random.rand(init.size)/100000.


mg = RasterModelGrid(100, 100, spacing=(1,1))
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05 + mg.node_x*0.05, at = 'node')
qs = mg.add_zeros('node', 'sediment __discharge')

#mg.set_fixed_value_boundaries_at_grid_edges(True, True, True, True)
mg.set_closed_boundaries_at_grid_edges(True, True, True, True) 

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'out', bottom = 'on', 
               left = 'on', top = 'off', right = 'off')
imshow_grid(mg,z, plot_name = 'Topographic Map of Synthetic Grid', var_name = 'Elevation', 
            var_units = 'm', grid_units = ('m','m'), cmap = 'jet')


#%% Exner Solver

def ExnerSolver(z, dqs_dx):
    
    return(z)
    
def div_qs(qs, grid):
    qs_link = grid.map_link_tail_node_to_link(qs)
    dqs_dx = grid.calc_flux_div_at_node(qs_link)
    
    return(dqs_dx)
    
def sed_disch(qs, grid,  drainage_area, dzdx, k_t=0.01, m=2, n=2):
    
    # $Q_s = k_t A^m S^n$
    # where Q_s = sediment discharge
    #       k_t = erodibility coefficient (choose arbitrarily)
    #       A = cell area
    #       S = slope
    #       m & n are empirical constants
    
    for node in grid.core_nodes:
        qs[node] = k_t * (drainage_area[node]**m) * (dzdx[node]**n)
       
    grid.at_node['sediment__discharge'] = qs
    
    return (qs, grid)    


#%% This is used to get ordered nodes for the sediment transport
def ordered(grid, outlet_id=None):
    
    if outlet_id == None:
        outlet_id = np.argmin(grid.at_node['topographic__elevation'])
   
    grid.set_watershed_boundary_condition_outlet_id(outlet_id, 
                                                    grid.at_node['topographic__elevation'], 
                                                    nodata_value=-9999.)
#### Change from FlowRouter to FlowDirector and FlowAccumulator
    fa = FlowAccumulator(grid, 
                         flow_director = FlowDirectorD8)
    fa.run_one_step()
    (da, q) = fa.accumulate_flow()
    ordered_nodes = grid.at_node['flow__upstream_node_order']
    
    return ordered_nodes, grid, da, q

#%% Calculate the slopes for slope-area plots

def calculate_slope(grid,z):
    dzdx = grid.calc_grad_at_link(z)
    grid.at_node['gradient'] = grid.map_max_of_inlinks_to_node(dzdx)
    dzdx = grid.at_node['gradient']
    
    return dzdx

#%%
ordered_nodes, mg, da, q = ordered(mg)
plt.figure()
drainage_plot(mg, 'drainage_area')

#%%

dzdx = calculate_slope(mg, z)

plt.figure()
plt.title('Slope-Area Plot for Initial Conditions')
plt.xlabel('Area (m$^2$)')
plt.ylabel('Slope (-)')
plt.loglog(mg.at_node['drainage_area'], dzdx, 'ko')
plt.xlim(10**(0)-10, 10**(4) + 5000)   
plt.ylim(10**(-4), 10**(-1)) 
plt.show()

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

slope = analytical_soln(mg)
drainage_area = mg.at_node['drainage_area'][mg.core_nodes]

plt.figure()
plt.loglog(drainage_area, slope, 'b-')
plt.title('Slope-Area Plot of Analytical Solution')
plt.xlabel('Area (m$^2$)')
plt.ylabel('Slope (-)')
plt.xlim(10**(0)-10, 10**(4) + 5000)   
plt.ylim(10**(-4), 10**(-1)) 
plt.show()


#%% Calculate sediment discharge and plot

qs, mg = sed_disch(qs, mg,  da, dzdx)

plt.figure()
imshow_grid(mg, qs, plot_name = 'Sediment Discharge at t = 0', var_name = 'Sediment Discharge', 
            var_units = r'$\frac{Q^3}{s}$', grid_units = ('m','m'), cmap = 'jet')


#%% Calculate sediment divergence and plot

divergence = div_qs(qs, mg)

plt.figure()
imshow_grid(mg, divergence, plot_name = 'Divergence of Sediment Discharge at t = 0', var_name = 'Sediment Flux', 
            var_units = r'$\frac{Q^2}{s}$', grid_units = ('m','m'), cmap = 'jet')



    