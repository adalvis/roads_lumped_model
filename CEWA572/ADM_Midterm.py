"""
Author: Amanda Manaster
Purpose: Solve advection-diffusion equation using the Crank-Nicolson 
         finite difference scheme
Date: 05/22/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

L = 50 #length of rod in meters
n = 49
dX = 1.0
x = np.linspace(0, L, n+1)

T = [20,100]
dT = 0.05
t = [1, 5]

k = 0.3333 #diffusivity in m^2/day 
u = 5.0 #advection velocity in m/day
c0 = 0 #mg/L
c1 = 100 #mg/L

D = (k*dT)/(dX**2)
S = (u*dT)/dX

#%% Part B of Midterm

# Analytical solution
c_ana = np.zeros((len(t),len(x)))

for j in range(len(t)):
    for i in range(len(x)):
        c_ana[j,i] = (c1/2)*(math.erfc((x[i]-u*t[j])/(2*np.sqrt(k*t[j]))) + \
                     np.exp(u*x[i]/k)*math.erfc((x[i]+u*t[j])/(2*np.sqrt(k*t[j]))))

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
for j in range(len(t)):
    color = ['m','g'] 
    plt.plot(x,c_ana[j,:], color = color[j], label = 't = %i day, analytical' % t[j] )
    
plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
#plt.savefig('C:/Users/Amanda/Desktop/PartB.png')

#%% Part C of Midterm

# Create A matrix
A = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    A[j+1, j] = -D-S/2
    A[j, j+1] = S/2-D
    
for i in range(len(x)):
    A[i,i] = 2+2*D
    
A[n,n] = 2+2*D    

# Create B matrix
B = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    B[j+1, j] = D+S/2
    B[j, j+1] = D-S/2
    
for i in range(len(x)):
    B[i,i] = 2-2*D
    
B[n,n] = 2-2*D    

# Create C matrix
g1 = c1
g2 = c0

C = np.zeros((len(x), 1))

C[0,0] = 2*g1*(D+S/2)
C[n, 0] = g2

C = np.matrix.transpose(C)

# Solve matrix equation and plot
plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')

for l in range(len(T)):
    c = np.zeros((T[l],n+1))

    for j in range(T[l]-1):
        inv = np.linalg.inv(A)
        ans = np.matrix.transpose((np.matmul(B,c[j,:])+C))
        c[j+1,:] = np.matrix.transpose(np.matmul(inv,ans))
    
    time = T[l]*0.05    
    color = ['k','c']    
        
    plt.plot(x, c[j,:], color = color[l], label = 't = %i day, approximate' % time)    

plt.plot(x, c_ana[0,:], color = 'm', label = 't = 1 day, analytical')
plt.plot(x, c_ana[1,:], color = 'g', label = 't = 5 day, analytical')
plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
#plt.savefig('C:/Users/Amanda/Desktop/PartC.png')
    
#%% Part E of Midterm

# Create A' matrix 
A_prime = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    A_prime[j+1, j] = -D/2
    A_prime[j, j+1] = -D/2
    
for i in range(len(x)):
    A_prime[i,i] = 1+D
    
A_prime[n,n] = 1+D    

# Create B' matrix
B_prime = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    B_prime[j+1, j] = D/2+S
    B_prime[j, j+1] = D/2
    
for i in range(len(x)):
    B_prime[i,i] = 1-S-D
    
B_prime[n,n] = 1-S-D    

# Create C' matrix
g1 = c1
g2 = c0

C_prime = np.zeros((len(x), 1))

C_prime[0,0] = g1*(D+S)
C_prime[n, 0] = g2

C_prime = np.matrix.transpose(C_prime)

# Solve matrix equation and plot
plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')

for l in range(len(T)):
    c_prime = np.zeros((T[l],n+1))

    for j in range(T[l]-1):
        inv_prime = np.linalg.inv(A_prime)
        ans_prime = np.matrix.transpose(np.matmul(B_prime,c_prime[j,:])+C_prime)
        c_prime[j+1,:] = np.matrix.transpose(np.matmul(inv_prime,ans_prime))
    

    time = T[l]*0.05    
    color = ['k--','c--'] 
            
    plt.plot(x, c_prime[j,:], color[l], label = 't = %i day, upgradient' % time)    
plt.plot(x, c[19,:], 'k', label = 't = 1 day, original')
plt.plot(x, c[99,:], 'c', label = 't = 5 day, original')
    
plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
#plt.savefig('C:/Users/Amanda/Desktop/PartE.png')   

#%% Part F of Midterm

# Define new kappa and D (for simplicity)
k_new = k +((-dX*u)/2) #diffusivity in m^2/day 

D_new = (k_new*dT)/(dX**2)

# Create new A' matrix
A_prime_new = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    A_prime_new[j+1, j] = -D_new/2
    A_prime_new[j, j+1] = -D_new/2
    
for i in range(len(x)):
    A_prime_new[i,i] = 1+D_new
    
A_prime_new[n,n] = 1+D_new    

# Create new B' matrix
B_prime_new = np.zeros((len(x),len(x)))
  
for j in range(len(x)-1):    
    B_prime_new[j+1, j] = D_new/2+S
    B_prime_new[j, j+1] = D_new/2
    
for i in range(len(x)):
    B_prime_new[i,i] = 1-S-D_new
    
B_prime_new[n,n] = 1-S-D_new    

# Create new C' matrix
g1 = c1
g2 = c0

C_prime_new = np.zeros((len(x), 1))

C_prime_new[0,0] = g1*(D_new+S)
C_prime_new[n, 0] = g2

C_prime_new = np.matrix.transpose(C_prime_new)

# Solve matrix equation and plot
plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')

for l in range(len(T)):
    c_prime_new = np.zeros((T[l],n+1))
    
    for j in range(T[l]-1):
        inv_prime_new = np.linalg.inv(A_prime_new)
        ans_prime_new = np.matrix.transpose(np.matmul(B_prime_new,c_prime_new[j,:])+C_prime_new)
        c_prime_new[j+1,:] = np.matrix.transpose(np.matmul(inv_prime_new,ans_prime_new))

    time = T[l]*0.05    
    color = ['k--*','c--*'] 
            
    plt.plot(x, c_prime_new[j,:], color[l], label = 't = %i day, upgradient, new diffusion' % time)    
plt.plot(x, c[19,:], 'k', label = 't = 1 day, original')
plt.plot(x, c[99,:], 'c', label = 't = 5 day, original')
    
plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
#plt.savefig('C:/Users/Amanda/Desktop/PartF.png')  