"""
Created on Wed Apr 18 13:10:39 2018

Updated 04/25/2018

Author: Amanda
"""

#import Python libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Forward Euler Method for du/dt = -u

t_s = 0
t_e = 10
n = 1000
dT = ((t_e-t_s)/n)
u_fwd = np.zeros(n)
t = np.linspace(0, 10, num = n)
u_fwd[0] = 1

u_analytical = np.exp(-t)

for i in range(n-1):
    u_fwd[i+1] = -u_fwd[i]*dT+ u_fwd[i]

plt.figure()    
plt.semilogx(t, u_analytical, 'k-')
plt.semilogx(t, u_fwd, 'y--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.legend()
plt.title('Forward Euler Method')
plt.plot()

#%% Backward Euler Method for du/dt = -u

t_s = 0
t_e = 10
n = 1000
dT = ((t_e-t_s)/n)
u = np.empty(n)
t = np.empty(n)
u[0] = 1
t[0] = 0
u_old = u[0]
t_old = t[0]

for i in range(n-1):
    
    t_new = t_old + dT
    
    dsol = 0.1
    tol = 0.01
    
    while dsol > tol:
        u_new = u_old - (u[i]-u_old*dT-u_old)/(dT-1)
        dsol = np.abs(u_new-u_old)
        u_old = u_new
        
    t_old = t_new
    u[i+1] = u_new
    t[i+1] = t_new

plt.figure()    
plt.semilogx(t, u_analytical, 'k-')
plt.semilogx(t, u, 'g--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('Backward Euler Method')
plt.plot()

#%% Corrected Euler Method for du/dt = -u

f = lambda u, t: -u

(t, u_cor) = rk2(f, 0, 10, 1000, 1) 

plt.figure() 
plt.semilogx(t, u_analytical, 'k-')
plt.semilogx(t, u_cor, 'c--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('Corrected Euler Method')
plt.plot()

#%%

f = lambda u, t: -u

(t, u_4) = rk4(f, 0,10, 1000, 1)

plt.figure()
plt.semilogx(t, u_analytical, 'k-')
plt.semilogx(t, u_4, 'm--')
plt.xlabel('Time (seconds)')
plt.ylabel('Function')
plt.title('4th Order Runge-Kutta Method')
plt.plot()
