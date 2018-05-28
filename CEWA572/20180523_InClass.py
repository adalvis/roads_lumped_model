"""
Author: Amanda
Date: 05/23/2018
"""
import numpy as np
import matplotlib.pyplot as plt

L = 50 #length of string
n = 100
dX = L/(n+1)
x = np.linspace(0, L, n+1)

T = 80
m = 1000
dT = T/m


c = 2 #m/s
s = ((c**2)*(dT**2))/dX**2

#%% Create A matrix
A = np.zeros((len(x),len(x)))

  
for j in range(len(x)-1):    
    A[j+1, j] = s
    A[j, j+1] = s
    
for i in range(len(x)):
    A[i,i] =2*(1-s)
    
A[n,n] = 2*(1-s)    

#%% Create B matrix
B = np.zeros((len(x), 1))

B = np.matrix.transpose(B)

#%% Create D matrix

D = np.zeros((len(x),1))

D = np.matrix.transpose(D)

#%% Solve equation
a_0 = 1
x_0 = 25
sigma = 2

u_0 = a_0*np.exp((-(x-x_0)**2/(2*sigma**2)))

u = np.zeros((m+1, n+1))

u[0,:] = u_0

plt.figure()

ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')

plt.plot(x, u_0)
plt.xlabel('X (m)')
plt.ylabel('Function')

#%%
for j in range(m):
    if j == 0:
        u[(j+1),:] = 0.5*np.matmul(A,u[j,:]) + 0.5*B + D
    else:    
        u[(j+1),:] = np.matmul(A, u[j,:]) + B - u[(j-1),:]

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.pcolor(u)
plt.xlabel('X (m)')
plt.ylabel('T (s)')
