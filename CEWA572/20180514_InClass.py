"""
Author: Amanda
Date: 05/14/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import math

L = 250 #length of rod in meters
n = 100
dX = L/(n+1)
x = np.linspace(0, L, n+1)

T = 365
m = 100
dT = T/m

t = [1*365,3.5*365,5*365]


k = 0.5 #diffusivity in m^2/day 
c = 0.05 #advection velocity in m/day


D = (k*dT)/(dX**2)
S = (c*dT)/dX

#%% Analytical solution

u0 = 1

u_ana = np.zeros((len(t),len(x)))

for j in range(3):
    for i in range(len(x)):
        u_ana[j,i] = (u0/2)*(math.erfc((x[i]-c*t[j])/(2*np.sqrt(k*t[j]))) + \
                     np.exp(c*x[i]/k)*math.erfc((x[i]+c*t[j])/(2*np.sqrt(k*t[j]))))

for j in range(3):
    plt.plot(x,u_ana[j,:])

plt.xlabel('X (m)');
plt.ylabel('Concentration');

#%% Create A matrix
A = np.zeros((len(x),len(x)))

  
for j in range(len(x)-1):    
    A[j+1, j] = D+S/2
    A[j, j+1] = D-S/2
    
for i in range(len(x)):
    A[i,i] = 1-2*D
    
A[n,n] = 1-2*D    

#%% Create B matrix

g1 = 1
g2 = 0

B = np.zeros((len(x), 1))

B[0,0] = g1*(D+S/2)
B[n, 0] = g2*(D-S/2)

B = np.matrix.transpose(B)

#%% Solve equation
u = np.zeros((n+1,m+1))

for j in range(n):
    u[(j+1),:] = np.matmul(A, u[j,:])+B
        
plt.plot(x, u[i,:])    
plt.plot(x, u_ana[0,:])