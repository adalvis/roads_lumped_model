"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Define functions used to calculate sediment transport on the grid
"""

import numpy as np
import matplotlib as mpl

from landlab.components import FlowAccumulator, FlowDirectorDINF

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#%% Functions needed for calculating sediment transport


# ordered() calculates the drainage area for each node and also 
#   returns the upstream to downstream node order
def ordered(grid, outlet_id=None):
    
    if outlet_id == None:
        outlet_id = np.argmin(grid.at_node['topographic__elevation'])
   
    grid.set_watershed_boundary_condition_outlet_id(outlet_id, grid.at_node['topographic__elevation'], 
                                                    nodata_value=-9999.)

    fa = FlowAccumulator(grid, flow_director = FlowDirectorDINF)
    fa.run_one_step()
    (da, q) = fa.accumulate_flow()
    
    ordered_nodes = grid.at_node['flow__upstream_node_order']
    ordered_nodes = ordered_nodes[::-1]
    
    return (ordered_nodes, grid, da)

# calculate_slope() determines the slope at each of the links and
#   maps the maximum slope of the inlinks to the node
def calculate_slope(grid, z):
    dzdx = grid.calc_grad_at_link(z)
    grid.at_node['gradient'] = grid.map_max_of_inlinks_to_node(dzdx)
    dzdx = grid.at_node['gradient']
    
    return (dzdx, grid)


# sed_disch() calculates sediment discharge used in finding 
#   the change in elevation of the surface
def sed_disch(qs, grid, ordered_nodes, drainage_area, dzdx, k_t=0.01, m=2, n=2):
    # $Q_s = k_t A^m S^n$
    # where Q_s = sediment discharge
    #       k_t = erodibility coefficient (choose arbitrarily)
    #       A = cell area
    #       S = slope
    #       m & n are empirical constants
    
    for node in ordered_nodes:
        qs[node] = k_t * (drainage_area[node]**m) * (dzdx[node]**n)
       
    grid.at_node['sediment__discharge'] = qs
    
    return (qs, grid)   

# div_qs() calculates the divergence of sediment discharge at
#   each node
def div_qs(qs, grid):
    
    qs_link = grid.map_link_head_node_to_link(qs)
    dqs_dx = grid.calc_flux_div_at_node(qs_link)
    
    return(dqs_dx)
    
# ExnerSolver() is the final step in determining how the topography
#   changes over time; both z and dzdt are solved for
def ExnerSolver(grid, z, dzdt, ordered_nodes, dqs_dx, dt):
    
    for node in ordered_nodes:
        dzdt[node] = -dqs_dx[node]
        z[node] = z[node]-dqs_dx[node]*dt
            
    return(z, dzdt)

# Q_out() updates the sediment discharge at every node
def Q_out(qs, dzdt, ordered_nodes, grid):
    
    for node in ordered_nodes:
        if node == 0:
            qs[node] = -dzdt[node] * grid.cell_area_at_node[node]
        else:
            qs[node] += -dzdt[node] * grid.cell_area_at_node[node]
        
    return(qs)
    
    
