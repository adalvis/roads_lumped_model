"""
Author: Amanda Manaster
Date: 09/18/2018
Purpose: Trying to create Exner solver
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from landlab.components import FlowRouter, DepressionFinderAndRouter
from landlab.plot.imshow import RasterModelGrid
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1

#%%

init = np.ones([10000])
surface = (init*12) + np.random.rand(init.size)/100000.


mg = RasterModelGrid(100, 100, spacing=(1,1))
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05 + mg.node_x*0.05, at = 'node')

mg.set_fixed_value_boundaries_at_grid_edges(True, True, True, True)
mg.set_closed_boundaries_at_grid_edges(False, False, False, False) 

plt.figure()
imshow_grid(mg,z)

def ExnerSolver(z, dqs_dx):
    
    return(z)
    
def div_qs(qs, order, grid):
    dqs = []
    dx = []
    for i in range(len(order)):
        dqs.append(qs[grid.node[i+1]]-qs[grid.node[i]])
        dx.append(grid.node[i+1]-grid.node[i])
        dqs_dx = dqs/dx
    return(dqs_dx)
    
def qs():
    
    # $Q_s = k_t A^m S^n$
    # where Q_s = sediment discharge
    #       k_t = erodibility coefficient (choose arbitrarily)
    #       A = cell area
    #       S = slope
    #       m & n are empirical constants
    
    
    
    return (qs)    


#%% Pay attention to ####
def get_ordered_cells(grid, outlet_id=None):
    """
    Written by: Sai & Erkan
    
    Runs Landlab's FlowRouter and DepressionFinderAndRouter to
    route flow. Also orders the cells in the descending order of
    channel length (upstream cell order).
    
    Parameters:
    ==========    
    grid: grid object
        RasterModelGrid
    outlet_id: int (Optional)
        Outlet id to be set

    Returns:
    =======
    ordered_cells: np.array(dtype=int)
        cells ordered in descending order of channel length
    grid: grid object
        updated RasterModelGrid
    """

    if outlet_id == None:
        outlet_id = np.argmin(grid.at_node['topographic__elevation'])
   
    grid.set_watershed_boundary_condition_outlet_id(outlet_id, 
                                                    grid.at_node['topographic__elevation'], 
                                                    nodata_value=-9999.)
#### Change from FlowRouter to FlowDirector and FlowAccumulator
    flw_r = FlowRouter(grid)
    flw_r.run_one_step()
    df = DepressionFinderAndRouter(grid)
    df.map_depressions()
    r = grid.at_node['flow__receiver_node'][grid.node_at_core_cell]
    R = np.zeros(grid.number_of_nodes, dtype=int)
    R[grid.node_at_core_cell] = r
    channel_length = np.zeros(grid.number_of_nodes, dtype=int)
    
    # Compute channel lengths for each node in the wtrshd (node_at_core_cell)
    for node in grid.node_at_core_cell:
        node_c = node.copy()
        while R[node_c] != node_c:
            channel_length[node] += 1
            node_c = R[node_c]
    grid.at_node['channel_length'] = channel_length
    
    # Sorting nodes in the ascending order of channel length
    # NOTE: length of ordered_nodes = grid.number_of_core_cells
    ordered_nodes = grid.node_at_core_cell[
        np.argsort(channel_length[grid.node_at_core_cell])]
    
    # Sorting nodes in the descending order of channel length
    ordered_nodes = ordered_nodes[::-1]
    dd = 1    # switch 2 for while loop
    count_loops = 0 # No. of loops while runs
    while dd:
        dd = 0
        count_loops += 1
        sorted_order = list(ordered_nodes)
        alr_counted_ = []
        for node_ in sorted_order:
            donors = []
            donors = list(grid.node_at_core_cell[np.where(r==node_)[0]])
            if len(donors) != 0:
                for k in range(0, len(donors)):
                    if donors[k] not in alr_counted_:
                        sorted_order.insert(donors[k], sorted_order.pop(sorted_order.index(node_)))
                        dd = 1    
            alr_counted_.append(node_)
        ordered_nodes = np.array(sorted_order)
    ordered_cells = grid.cell_at_node[ordered_nodes]
    
    return ordered_cells, ordered_nodes, grid

#%% Calculate the slopes for slope-area plots


def calculate_slope(grid,z):
    dzdx = grid.calc_grad_at_link(z)
    grid.at_node['gradient'] = grid.map_max_of_inlinks_to_node(dzdx)
    dzdx = grid.at_node['gradient']
    
    return dzdx


#%%
ordered_cells, ordered_nodes, mg = get_ordered_cells(mg)
dzdx = calculate_slope(mg, z)

plt.figure()
plt.loglog(mg.at_node['drainage_area'], dzdx, 'ko')
plt.xlim(10**(0)-10, 10**(4) + 5000)   
plt.ylim(10**(-4), 10**(-1)) 


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
plt.loglog(drainage_area, slope, 'b-')
plt.show()







    