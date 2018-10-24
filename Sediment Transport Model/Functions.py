"""
Author: Amanda Manaster
Date: 10/17/2018
Purpose: Define functions used to calculate sediment transport on the grid
"""

import numpy as np
import matplotlib as mpl

from landlab.components import FlowAccumulator, FlowDirectorD8, DepressionFinderAndRouter

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#%% Functions needed for calculating sediment transport

fa = FlowAccumulator(mg, flow_director = FlowDirectorD8)
df = DepressionFinderAndRouter(mg)

# ordered() calculates the drainage area for each node and also 
#   returns the upstream to downstream node order
def ordered(grid, fa, outlet_id=None):
    
#    if outlet_id == None:
#        outlet_id = np.argmin(grid.at_node['topographic__elevation'])
#        grid.set_watershed_boundary_condition_outlet_id(outlet_id, grid.at_node['topographic__elevation'], 
#                                                        nodata_value=-9999.) 
    
    fa.run_one_step()
    
    da = grid.at_node['drainage_area']
    
    ordered_nodes = grid.at_node['flow__upstream_node_order']
    ordered_nodes = np.flipud(ordered_nodes)
    
    return (ordered_nodes, grid, da)

# calculate_slope() determines the slope at each of the links and
#   maps the maximum slope of the inlinks to the node
def calculate_slope(grid, z):
    dzdx = grid.calc_grad_at_link(z)
    dzdx = grid.map_downwind_node_link_max_to_node(dzdx)
    
    return (dzdx)


# sed_disch() calculates sediment discharge used in finding 
#   the change in elevation of the surface
def sed_disch(qs, qs_in, grid, z, ordered_nodes, drainage_area, dzdx, dzdt, 
              k_t=0.01, m=2, n=2):
    # $Q_s = k_t A^m S^n$
    # where Q_s = sediment discharge
    #       k_t = erodibility coefficient (choose arbitrarily)
    #       A = cell area
    #       S = slope
    #       m & n are empirical constants
    
    flow_receiver = grid.at_node['flow__receiver_node']
    
    for node in ordered_nodes:
        qs[node] = qs_in[node] + (k_t * (drainage_area[node]**m) * (dzdx[node]**n))
        
        qs_in[flow_receiver[node]] += qs[node]
        qs[node] = qs_in[node] - dzdt[node] * grid.cell_area_at_node[node]
        
    
    return (qs, qs_in, grid)   

# div_qs() calculates the divergence of sediment discharge at
#   each node
def div_qs(qs, z, grid):
    
    qs_link = grid.map_value_at_max_node_to_link(z, qs)
    dqs_dx = grid.calc_flux_div_at_node(qs_link)
    
    return(dqs_dx, qs_link)
    
# ExnerSolver() is the final step in determining how the topography
#   changes over time; both z and dzdt are solved for
def ExnerSolver(grid, z, dzdt, ordered_nodes, dqs_dx, dt):
    
    for node in ordered_nodes:
        dzdt[node] = -dqs_dx[node]
        z[node] = z[node] - dqs_dx[node]*dt
            
    return(z, dzdt)

## Q_out() updates the sediment discharge at every node
#def Q_out(qs, qs_in, dzdt, ordered_nodes, grid):
#    
#    for node in ordered_nodes:
#        qs[node] = qs_in[node] - dzdt[node] * grid.cell_area_at_node[node]
#        
#    return(qs)   
    
    
