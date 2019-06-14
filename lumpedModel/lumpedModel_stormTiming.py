"""
Author: Amanda Manaster
Date: 06/10/2019
Purpose: Lumped model of road surface/sediment transport.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

#%% 
"""
Let's think about fines storage.

dS_f = max(u*n*A - q_s*t_s, 0)

where:
    S_f = storage of fine sediment in m^3
    u = pumping of fines w/ each truck pass; 0.05 mm
    n = # of truck passes; constant
    A = representative area of road in m^2
    t_s = storm duration in s
    q_s = mean rate of sediment transport during a storm, assuming steady-state runoff in m^2/s
       
Gover's equation (Istanbulluoglu et al. 2003):
    q_s = 10^(-4.348)/(rho_s*d50^0.811)*(tau-tau_c)^2.457
    
    *Note: This equation is calibrated for sediments that are between 0.058 and 1.098 mm.*

more variables:
    tau = shear stress; rho_w*g*H*S
    tau_c = critical shear stress; see 
    rho_w = density of water; 1000 kg/m^3
    g = gravity; 9.81 m/s^2
    H = depth of water; r*t_s
    S = slope
    rho_s = density of sediment; 2650 kg/m^3
    r = runoff rate in m/s
    d50 = median grain size of fines
   

"""
A = 3.66
S = 0.058
time = 0 
n = 0
r = 0
month = []
H = []
t = []
len_s = []

model_duration = np.linspace(0,179,180)
model_end = 4320 #hours

while time < model_end:
    T_b = np.random.uniform(12,240)
    T_r = np.random.exponential(1.7)
    n += 20
    r = np.random.uniform(10,25)
    
    len_s.append(T_r)
    
    H.append(T_r*r)
    t.append(time)
    
    time += T_b + T_r

df = pd.DataFrame()
df['time'] = t
df['day'] = np.divide(t,24).astype('int64')
df['water_depth'] = H
df['storm_length'] = len_s

day0 = datetime(2018, 10, 1)
df.loc[:,'dt'] = [day0+timedelta(hours=time) for time in df.time]

df = df.set_index(pd.DatetimeIndex(df.dt))

df2 = df.resample('D').mean().fillna(0)


ticklabels = [item.strftime('%b %d') for item in df2.index[::10]]
fig, ax = plt.subplots(figsize=(9, 6))
df2.plot.bar(y='water_depth', ax=ax, legend=False)
plt.xlabel('Date')
plt.ylabel('Water depth (mm)')
plt.xticks(np.arange(0,10*len(ticklabels),10), ticklabels, rotation=35)
plt.show()