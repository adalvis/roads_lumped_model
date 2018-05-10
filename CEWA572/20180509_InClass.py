"""
Created on Wed May  9 13:09:43 2018

Author: Amanda
"""
import numpy as np
import matplotlib.pyplot as plt

L = 1
T = 12000
n = 39
m = 600

dT = T/m
dX = L/(n+1)

k = 10**(-5) 

x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m)

s = (k*dT)/(dX**2)

#%% Create A matrix
A = np.zeros((len(x),len(x)))

  
for j in range(len(x)-1):    
    A[j+1, j] = s
    A[j, j+1] = s
    
for i in range(len(x)):
    A[i,i] = 1-2*s
    
A[n,n] = 1-s    


#v1 = np.ones(n, 1)*(1-2*s)
#v2 = np.ones(n-1,1) * s 
#A1 = np.diagonal(v1, 0) 
#A2 = np.diagonal(v2, 1)
#A3 = np.diagonal(v2, -1)
#A = A1+A2+A3    

#%% Create B matrix

g1 = 1
g2 = 2

B = np.zeros((len(x), 1))

B[0,0] = s*g1
B[n, 0] = s*dX*g2

B = np.matrix.transpose(B)

#%% Initial conditions

U = np.zeros((len(x), len(t)))

for i in range(len(x)):
    U[i,0] = 1+2*x[i]+np.sin(2*np.pi*x[i])
    
u = U[:,0]



for j in range(len(t)-1):
    U_ans = np.matmul(u, A)+B
    U[:, j+1] = U_ans
    u = U_ans
    
    
plt.pcolor(np.matrix.transpose(U))


