"""
Author: Amanda Manaster
Date: 11/06/2018
Purpose: Create plots of x-offset vs. y-offset for a series of slopes
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Full set of slopes

theta = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40])
x = np.linspace(0.01, 10, 100)
y = np.zeros((len(theta), len(x)))

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')

for i in range(len(theta)):
    y[i, :] = x*np.tan(np.radians(theta[i])) 
    
    color = ['#1A000C','#350018','#4F0024','#4F0024','#6A0030','#84003C','#9E0048','#B90054',\
             '#D30060','#D92074','#DE4088','#E4609C','#E980B0','#EF9FC3']        
    plt.plot(x, y[i,:], color = color[i], label = r'$ \theta $ = %i$^{\circ}$' % theta[i])
    
plt.xlabel('Horizontal Offset (m)')
plt.ylabel('Vertical Offset (m)')
plt.legend(loc = 'best', ncol = 2)  

#%% Subset of slopes relevant to field sites
theta_sub = np.array([2, 3, 4, 5, 6, 7])
x_sub = np.linspace(0.01, 10, 100)
y_sub = np.zeros((len(theta_sub), len(x_sub)))


plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')

for j in range(len(theta_sub)):
    y_sub[j, :] = x_sub*np.tan(np.radians(theta_sub[j])) 
    
    color_sub = ['#4F0024','#4F0024','#6A0030','#84003C','#9E0048','#B90054']        
    plt.plot(x_sub, y_sub[j,:], color = color_sub[j], label = r'$ \theta $ = %i$^{\circ}$' % theta_sub[j])
    
plt.xlabel('Horizontal Offset (m)')
plt.ylabel('Vertical Offset (m)')
plt.legend(loc = 'best', ncol = 1)  