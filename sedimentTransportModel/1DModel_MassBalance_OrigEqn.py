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

n = 2.
m = 2.
U = 5e-5 #uplift rate m/year
k_t = 0.1 #erosion constant

L = 10e3 #length of river reach m
na = 100 #number of data points
dx = L/na #spacing between data points
a = np.arange(0, L, dx) #array of specific catchment areas ranging from 1 to 10e3

init = 1000 #elevation
s0 = 0.0007 #initial slope
z = np.linspace(init, init - s0*(na*dx), na) #initialize elevation array

#plot initial river profile
plt.figure(1)
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.plot(a, z)
plt.xlabel('Specific Catchment Area (m)')
plt.ylabel('Elevation (m)')

#calculate steady state slope
slope = np.zeros(len(a))
slope = ((U/k_t)**(1/n)) * (a)**((1-m)/n)

#plot steady state slope-area
plt.figure(2)
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.loglog(a, slope, 'b-')
plt.xlabel('Specific Catchment Area (m)')
plt.ylabel('Slope (-)')

#initialize arrays used later
qs_out = np.zeros(na) #outgoing sediment
qs_in = np.zeros(na) #incoming sediment
dqsdx = np.empty(na) #divergence of sediment flux
s = np.zeros(na) #local slope

#look at how the profile evolves through the years
T = [0, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000] #years

#timestep is small
dt = 0.1 #fraction of years
t = 0

plt.figure(3)
plt.figure(4)
#loop through years to plot
for l in range(len(T)):
    #loop through model driver    
    while t < T[l]:  
        #calculate initial outgoing sediment
        if t == 0:
            qs_out = (k_t)*(a**m)*(s0**n)
        else:
            qs_out = (k_t)*(a**m)*(s**n)
        #look at each point
        for i in range(na):
            #calculate local slope at each point
            if i == 0:
                s[i] = (z[i+1]-z[i])/dx #forward difference @ upstream
            elif i == na-1:    
                s[i] = (z[i]-z[i-2])/(2*dx) #backward difference @ downstream
            else:    
                s[i] = (z[i+1]-z[i-1])/(2*dx) #central difference everywhere else
            
            #divergence of sediment flux at upstream and downstream nodes 
            #    has to be 0 to keep system from blowing up
            if i == 0 or i == na-1:
                dqsdx[i] = 0 
            else:
                dqsdx[i] = (qs_out[i] - qs_in[i])/(dx)
                      
            #route the new outgoing sediment to be the next data point's 
            #    incoming sediment
            if i != na-1:  
                qs_in[i+1] = qs_out[i]       
        
        #calculate new elevation!
        z0 = z.copy()
        z = z0 - dqsdx*dt
        
        print(t)
        t+=dt #increase time
   
    #plot updated elevation profile
    plt.figure(3)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
                   left = 'on', top = 'on', right = 'on')
    plt.plot(a, z, label = r'%i years' % T[l])
    plt.xlabel('Specific Catchment Area (m)')
    plt.ylabel('Elevation (m)')
    plt.legend(loc = 'best')
    plt.show()
 
    #plot slope-area for t = 3000 years in model run and steady state solution
    plt.figure(4)
    ax2 = plt.gca()
    ax2.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
                   left = 'on', top = 'on', right = 'on')
    plt.loglog(a, slope, 'b-')
    plt.loglog(a, abs(s), 'o', label = r'%i years' % T[l])
    plt.xlabel('Specific Catchment Area (m)')
    plt.ylabel('Slope (-)')
    plt.legend(loc = 'best')
    plt.show()

#for keeping track of how long the model runs
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) 