"""
Created on Thu May 11 16:14:05 2017

Author: Amanda
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rnd

#import necessary Landlab components
from landlab.components import LinearDiffuser, FlowDirectorMFD
    
#import Landlab utilities
from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from landlab.plot.drainage_plot import drainage_plot


##set boundary conditions
#mg.set_fixed_value_boundaries_at_grid_edges(True, False, True, False)
#mg.set_closed_boundaries_at_grid_edges(False, True, False, True)              

#%% Create erodible grid

tire_1 = 16 #x-position of one tire
tire_2 = 24 #x-position of other tire

out_1 = [15,17] #x-positions of the size cells of the first tire
out_2 = [23,25] #x-positions of the size cells of the other tire

back_tire_1 = [] #initialize the back of tire recovery for first tire
back_tire_2 = [] #initialize the back of tire recovery for other tire

mg_erode = RasterModelGrid(355,51,0.225) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
init_elev = np.zeros(mg_erode.number_of_nodes, dtype = float) #initialize the elevation grid
z_erode = mg_erode.add_field('topographic__elevation', init_elev, at = 'node') #create the topographic__elevation field

mg_erode.set_closed_boundaries_at_grid_edges(True, True, True, True) 

road_peak = 20 #peak crowning height occurs at this x-location
up = 0.0067 #rise of slope from ditchline to crown
down = 0.0035 #rise of slope from crown to hillslope

for g in range(0,355): #loop through road length
    elev = 0 #initialize elevation placeholder
    
    for h in range(0, 51): #loop through road width
        z_erode[g*51 + h] = elev #update elevation based on x & y locations
        
        if h == 0 or h == 4:
            elev = 0
        elif h == 1 or h == 3:
            elev = -0.5715
        elif h == 2:
            elev = -0.762
        elif h < road_peak and h > 3: #update latitudinal slopes based on location related to road_peak
            elev += up
        else:
            elev -= down

z_erode += mg_erode.node_y*0.05 #add longitudinal slope to road segment

#%% Time to try a basic model!

#get node IDs for the important nodes
tire_track_1 = mg_erode.nodes[:, tire_1]
tire_track_2 = mg_erode.nodes[:, tire_2]
out_tire_1 = mg_erode.nodes[:, out_1]
out_tire_2 = mg_erode.nodes[:, out_2]

back_tire_1.append(mg_erode.nodes[0, tire_1])
back_tire_2.append(mg_erode.nodes[0, tire_2])

for k in range(0,354):
    back_tire_1.append(mg_erode.nodes[k+1, tire_1])
    back_tire_2.append(mg_erode.nodes[k+1, tire_2])

#initialize truck pass and time arrays
truck_pass = []
time = []

#define how long to run the model
model_end = 10 #days

#initialize LinearDiffuser component
lin_diffuse = LinearDiffuser(mg_erode, linear_diffusivity = 0.0001)

for i in range(0, model_end): #loop through model days
    #initialize/reset the times for each loop
    t_recover = 0
    t_pass = 0
    t_total = 0

    while t_total <=24:
        if t_total < 4:
            T_B_morning = rnd.expovariate(1/4)
            time.append(t_total+24*i)
            truck_pass.append(0)
            t_recover += T_B_morning
        elif t_total >= 4 and t_total <= 15:
            t_b = rnd.expovariate(1/2.2)
            z_erode[tire_track_1] -= 0.001
            z_erode[tire_track_2] -= 0.001
            z_erode[out_tire_1] += 0.0004
            z_erode[out_tire_2] += 0.0004
            z_erode[back_tire_1] += 0.0002
            z_erode[back_tire_2] += 0.0002
            time.append(t_total+24*i)
            truck_pass.append(1)
            t_pass += t_b              
        elif t_total > 15:
            T_B_night = rnd.expovariate(1/9)
            lin_diffuse.run_one_step(T_B_night)
            time.append(t_total+24*i)
            truck_pass.append(0)
            t_recover += T_B_night                
        
        t_total = t_pass + t_recover
        
plt.figure(figsize = (4,10))
imshow_grid(mg_erode, z_erode, var_name = 'Elevation', 
            var_units = 'm',grid_units = ('m','m'), cmap = 'gist_earth')
plt.title('Road Surface Elevation', fontweight = 'bold')   

#z_erode[outlet_id_1] -= 0.05
#z_erode[outlet_id_2] -= 0.05
#z_erode[outlet_id_3] -= 0.05
#z_erode[100] -= 0.05     
#%% 

new_x_extent=[0, 51]
new_y_extent=[0, 20]

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
        mg0_ind[k]= int(51*(start_row+i)+start_col+j)
        x = int(mg0_ind[k])
        mg0_z[k]=z_erode[x]
        k += 1

mg0=RasterModelGrid((nrows,ncols), spacing=(1.,1.))
mg0.add_field('node','topographic__elevation', mg0_z)

plt.figure()
imshow_grid(mg0, 'topographic__elevation', limits=(0, np.max(mg0_z)), plot_name='Elevation (m)')
plt.show()

#%%

fd = FlowDirectorMFD(mg0)
fd.run_one_step()

plt.figure()
drainage_plot(mg0)
#plt.savefig('C:/Users/Amanda/Desktop/After.png')
plt.show()