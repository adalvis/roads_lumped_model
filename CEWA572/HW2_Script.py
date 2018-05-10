"""
Author: Amanda Manaster
Date: 20180506
Purpose: Solve differential equation using (a) forward, (b) backward, (c) 2nd order RK, and (d) 4th order RK
         schemes calling on functions in HW2_Functions.py.

"""
#%% Import Python libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

#%% Forward Euler for du/dt = -30u

f = lambda u, t: -30*u

(t, u_fwd) = fwd(f, 0, 1.5, 100, 1/3) 

u_analytical = (1/3)*np.exp(-30*t)

plt.figure() 
plt.plot(t, u_analytical, 'k-')
plt.plot(t, u_fwd, 'y--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('Forward Euler Method', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/FwdEuler.png', bbox_inches = 'tight')
plt.plot()

#%% Backward Euler for du/dt = -30u

f = lambda u, t: -30*u
df = lambda u, t: -30

(t, u_bwd) = bwd(f, df, 0, 1.5, 100, 1/3, 0.01) 

u_analytical = (1/3)*np.exp(-30*t)


plt.figure()    
plt.plot(t, u_analytical, 'k-')
plt.plot(t, u_bwd, 'g--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('Backward Euler Method', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/BwdEuler.png', bbox_inches = 'tight')
plt.plot()

#%% Second-Order Runge Kutta for du/dt = -30u

f = lambda u, t: -30*u

(t, u_cor) = rk2(f, 0, 1.5, 100, 1/3) 

plt.figure() 
plt.plot(t, u_analytical, 'k-')
plt.plot(t, u_cor, 'c--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('2nd Order Runge-Kutta Method', fontweight = 'bold')
plt.savefig('C://Users/Amanda/Desktop/RK2.png', bbox_inches = 'tight')
plt.plot()

#%% Fourth-Order Runge Kutta for du/dt = -30u

f = lambda u, t: -30*u

(t, u_4) = rk4(f, 0, 1.5, 100, 1/3)

plt.figure()
plt.plot(t, u_analytical, 'k-')
plt.plot(t, u_4, 'm--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('4th Order Runge-Kutta Method', fontweight = 'bold')
#plt.savefig('C://Users/Amanda/Desktop/RK4.png', bbox_inches = 'tight')
plt.plot()

