"""
Author: Amanda Manaster
Date: 06/07/2019
Purpose: Lumped model of road surface/sediment transport.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% 
"""
Let's think about fines storage.

dS_f = max(u*n*A - q_s*t_s, 0)

where:
    S_f = storage of fine sediment in m^3
    u = pumping of fines w/ each truck pass; uniform distribution btwn 0.01 and 0.1 mm
    n = # of truck passes; exponential
    A = representative area of road in m^2
    t_s = storm duration in s
    q_s = mean rate of sediment transport during a storm, assuming steady-state runoff in m^3/s
    
more variables:
    q = runoff rate in m/s
    d50 = median grain size of fines
    S = slope
    V_bar = depth averaged flow velocity
    
Gover's equation:
    T_c = 86.7*(S*V_bar-0.005)/sqrt(d50)*q 
    
    *Note: This equation is calibrated for sediments that are between 0.058 and 0.218 mm.*

"""

t_s = 7200 #sec
A = 3.66 #m^2
S = 0.058 #-


day = 24
morn = 4
eve = 15
time = []
truck_pass = []
fines = []
model_duration = np.linspace(0,10,11)
model_end = 10 #days

for step in model_duration:
    t_total = 0
    t_recover = 0
    t_pass = 0
    
    while t_total < day:
    # average: 5 trucks per 11 hours = 1 truck per 2.2 hours
    # 2.2 hours is mean time between truck passes
    # lambda = 1/mean = 1/2.2
    # exponential wants 1/lamba = 2.2
        if (t_total < morn) or (t_total > eve):
            T_B = 4 if t_total < morn else 9
            time.append(t_total + (day*step))
            truck_pass.append(0)
            fines.append(0)
            t_recover += T_B    
        else:
            t_b = np.random.exponential(2.2)
            time.append(t_total + (day*step))
            truck_pass.append(1)
            fines.append(np.random.uniform(0.00001, 0.0001))
            t_pass += t_b
                
        t_total = t_pass + t_recover
       
    n = truck_pass
    u = fines
    S_f_supp = np.multiply(n,u)*A
    
x_axis = np.linspace(0, model_end*24, model_end+1)
a = [0,1]
    
plt.figure(figsize = (12, 5))
plt.bar(time, truck_pass, color = 'r', edgecolor = 'k')
plt.xticks(x_axis, np.linspace(0,model_end+1,model_end+2, dtype = int))
plt.yticks(a, ('No','Yes'))
plt.xlim(0,model_end*24)
plt.ylim(0,1.1)
plt.xlabel('Time (Days)')
plt.ylabel('Truck Pass?')
#plt.savefig('C://Users/Amanda/Desktop/TruckPass_YN.png', bbox_inches = 'tight')
plt.show() 

 
plt.figure(figsize = (7, 5))
plt.plot(time, np.cumsum(fines), 'b-')
plt.xticks(x_axis, np.linspace(0,model_end+1,model_end+2, dtype = int))
plt.xlim(0,model_end*24)
plt.xlabel('Time (Days)')
plt.ylabel('Cumulative depth of fines generated (m)')
plt.show()