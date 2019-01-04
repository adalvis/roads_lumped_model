"""
Author: Amanda Manaster
Date: 11/06/2018
Purpose:
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

from datetime import datetime
start_time = datetime.now()

#%%

n = 2.
m = 2.
U = 5e-5
k_t = 0.01

L = 10e3
na = 10
dx = L/na
a = np.arange(1, L, dx)

init = 515
s0 = 5e-4
z = np.linspace(init, init - s0*(na*dx), na)

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.plot(a, z)
plt.xlabel('Specific Catchment Area (m)')
plt.ylabel('Elevation (m)')




slope = np.zeros(len(a))
slope = ((U/k_t)**(1/n)) * (a)**((1-m)/n)

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.loglog(a, slope, 'b-')
plt.xlabel('Specific Catchment Area (m)')
plt.ylabel('Slope (-)')

#%%
qs_out = np.zeros(na)
qs_in = np.zeros(na)
dqsdx = np.empty(na)
s = np.zeros(na)

T = [0, 10, 50, 100, 500, 1000, 10000] #years
dt = 0.005 #fraction of years
t = 0

#%%
plt.figure()
for l in range(len(T)):

    while t < T[l]:  
    
        for i in range(na):
            
            if i == 0:
                s[i] = (z[i+1]-z[i])/dx
            elif i == na-1:    
                s[i] = (z[i]-z[i-2])/(2*dx)
            else:    
                s[i] = (z[i+1]-z[i-1])/(2*dx)
                
            qs_out[i] = (k_t)*(a[i]**m)*(s[i]**n)
            
            if i == 0 or i == na-1:
                dqsdx[i] = 0
            else:
                dqsdx[i] = (qs_out[i] - qs_in[i])/dx
                   
            qs_out[i] = qs_in[i] + dqsdx[i]*dx
            
            if i != 0 and i != na-1:  
                qs_in[i+1] = qs_out[i]       
            
        z0 = z.copy()
        z = z0 - dqsdx*dt
        
        print(t)
        t+=dt

    plt.plot(a,z)


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) 