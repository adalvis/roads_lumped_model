"""
Author: Amanda Manaster
Date: 11/08/2018
Purpose: Solve the Exner equation in 1-D
"""
#import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#set fonts
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#for keeping track of how long the model runs
from datetime import datetime
start_time = datetime.now()

#%%

n = 2.
m = 2.
U = 5e-5 #uplift rate m/year
k_t = 0.1 #erosion constant

L = 10e3 #length of river reach m
na = 100 #number of data points
dx = L/na #spacing between data points
x = np.arange(0, L, dx) #array of specific catchment areas ranging from 1 to 10e3

dt = 0.1
t = np.linspace(0, 8999, 9000/0.1) #years
T = np.array([0, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8999])*10


init = 1000 #elevation
s0 = 0.0006 #initial slope
s = np.zeros((len(x), len(t))) #local slope
s[:,0] = s0


z = np.zeros((len(x), len(t))) 
z[:, 0] = np.linspace(init, init - s0*(na*dx), na)#initialize elevation array

#plot initial river profile
plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.plot(x, z[:,0])
plt.xlabel('Specific Catchment Area (m)')
plt.ylabel('Elevation (m)')

#%%

#calculate steady state slope
slope = np.zeros(len(x))
slope = ((U/k_t)**(1/n)) * (x)**((1-m)/n)


#plot steady state slope-area
plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.loglog(x, slope, 'b-')
plt.xlabel('Specific Catchment Area (m)')
plt.ylabel('Slope (-)')

#%%


#loop through years to plot
for k in range(0, len(t)-1):
    
    for i in range(0,len(x)):
        
        if i == na-1 or i == 0:    
            z[i, k+1] = z[i, k]
            
            if i == 0:
                s[i, k] = (z[i+1,k]-z[i,k])/(dx)
            else:
                s[i, k] = (z[i,k]-z[i-1,k])/(dx)
                
        else:
            z[i, k+1] = z[i, k] - dt*((2*k_t)*(x[i]*((z[i+1,k] - z[i-1,k])/(2*dx))**2 + \
             (x[i])**2*((z[i+1, k] - z[i-1,k])/(2*dx))*((z[i+1, k] + z[i-1,k] - 2*z[i,k])/(dx)**2)))
            
            s[i, k] = (z[i+1,k]-z[i-1,k])/(2*dx)

        
    print(t[k])


#%%    
plt.figure()
for c in range(0,len(T)):    
    #plot updated elevation profile
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
                   left = 'on', top = 'on', right = 'on')
    plt.plot(x, z[:,T[c]], label = r'%i years' % T[c])
    plt.xlabel('Specific Catchment Area (m)')
    plt.ylabel('Elevation (m)')
    #plt.legend(loc = 'best')
    plt.show()

#plot slope-area for t = 3000 years in model run and steady state solution
plt.figure()
for c in range(0,len(T)): 
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
                   left = 'on', top = 'on', right = 'on')
    plt.loglog(x, slope, 'b-')
    plt.loglog(x, abs(s[:, T[c]]), 'o')
    plt.xlabel('Specific Catchment Area (m)')
    plt.ylabel('Slope (-)')
    
    plt.show()

#for keeping track of how long the model runs
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) 