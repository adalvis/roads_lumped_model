"""
Created on Thu May 11 16:14:05 2017

Author: Amanda
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#import necessary Landlab components
from landlab.components import LinearDiffuser, FlowDirectorMFD
    
#import Landlab utilities
from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

    
#%%
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#initialize a surface that has an elevation of 12m
init = np.ones([10000])
surface = init*12

#create a 100x100 grid and add a slanted surface elevation
mg = RasterModelGrid(100, 100, spacing = (1,1))
z = mg.add_field('topographic__elevation', surface + mg.node_y*0.05, at = 'node')

#set boundary conditions
mg.set_fixed_value_boundaries_at_grid_edges(True, False, True, False)
mg.set_closed_boundaries_at_grid_edges(False, True, False, True)              

#use LinearDiffuser to erode the core nodes; this creates a parabolic shape
lin_diffuse = LinearDiffuser(mg, linear_diffusivity = 0.8)
nt = int(2000 // 100)
for i in range(nt):
    lin_diffuse.run_one_step(100)
    z[mg.core_nodes] += -0.005 * 100  # erode
    
outlet_id = mg.core_nodes[np.argmin(mg.at_node['topographic__elevation'][mg.core_nodes])]                     
mg.set_watershed_boundary_condition_outlet_id(outlet_id, z)    
        
z[outlet_id] -= 0.05

#plot the grid showing its topographic elevation
imshow_grid(mg, 'topographic__elevation', var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth', 
            limits= (5.5, 17))
plt.title('Topographic Elevation, Slope = 0.05' , fontweight = 'bold')
plt.show()

#%% 

new_x_extent=[45, 55]
new_y_extent=[0, 10]

# Extract range of rows and columns to extract
rows=[int(np.round(new_y_extent[0]/1)),int(np.round(new_y_extent[1]/1))] # rows to extract from mg
cols=[int(np.round(new_x_extent[0]/1)),int(np.round(new_x_extent[1]/1))] # columns to extract from mg

start_row=rows[0]
end_row=rows[1]
start_col=cols[0]
end_col=cols[1]

ncols=end_col-start_col # number of rows in new grid
nrows=end_row-start_row # number of columns in new grid

new_grid_size=[nrows*ncols]

mg0_ind=np.zeros(new_grid_size)
mg0_z=np.zeros(new_grid_size)

k = 0

for i in range(0,nrows):
    for j in range(0,ncols):
        mg0_ind[k]= int(100*(start_row+i)+start_col+j)
        x = int(mg0_ind[k])
        mg0_z[k]=z[x]
        k += 1

mg0=RasterModelGrid((nrows,ncols), spacing=(1.,1.))
mg0.add_field('node','topographic__elevation', mg0_z)

imshow_grid(mg0, 'topographic__elevation', limits=(0, np.max(mg0_z)), plot_name='Elevation (m)')
plt.show()

#%%

fd = FlowDirectorMFD(mg0)
fd.run_one_step()

drainage_plot(mg0)
#plt.savefig('C:/Users/Amanda/Desktop/After.png')
plt.show()