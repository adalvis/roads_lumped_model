"""
Author: Amanda
Date: 05/20/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import math

L = 50 #length of rod in meters
n = 49
dX = L/(n+1)
x = np.linspace(0, L, n+1)

T = 50
dT = 0.05

t = [1, 5]


k = 0.5 #diffusivity in m^2/day 
u = 5.0 #advection velocity in m/day
c0 = 0 #mg/L
c1 = 100 #mg/L


D = (k*dT)/(dX**2)
S = (u*dT)/dX

#%% Analytical solution - Part B of Midterm

c_ana = np.zeros((len(t),len(x)))

for j in range(len(t)):
    for i in range(len(x)):
        c_ana[j,i] = (c1/2)*(math.erfc((x[i]-u*t[j])/(2*np.sqrt(k*t[j]))) + \
                     np.exp(u*x[i]/k)*math.erfc((x[i]+u*t[j])/(2*np.sqrt(k*t[j]))))

for j in range(len(t)):
    plt.plot(x,c_ana[j,:])

plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')

#%% Create A matrix
A = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    A[j+1, j] = -D-S/2
    A[j, j+1] = S/2-D
    
for i in range(len(x)):
    A[i,i] = 2+2*D
    
A[n,n] = 2+2*D    

#%% Create B matrix
B = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    B[j+1, j] = D+S/2
    B[j, j+1] = D-S/2
    
for i in range(len(x)):
    B[i,i] = 2-2*D
    
B[n,n] = 2-2*D    
#%% Create C matrix
g1 = c1
g2 = c0

C = np.zeros((len(x), 1))

C[0,0] = g1*(D+S/2)
C[n, 0] = g2*(D-S/2)

C = np.matrix.transpose(C)

#%% Solve equation
c = np.zeros((n+1,T))

for j in range(n):
    inv = np.linalg.inv(A)
    c[(j+1),:] = np.matmul([np.matmul(c[j,:], B) + C], inv)
            
plt.plot(x, c[i,:])    
plt.plot(x, c_ana[0,:])