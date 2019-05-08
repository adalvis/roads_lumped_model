"""
Author: Amanda Manaster
Purpose: Solve advection-dispersion-adsorption-degradation equation using the FTCS
         finite difference scheme
Date: 11/06/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

L = 250 #length of rod in meters
n = 249
dX = 1.0 #meters
x = np.linspace(0, L, n+1)

T = np.array([20*365,70*365,100*365])
dT = 0.05
t_year = np.array([1, 3.5, 5]) #YEARS
t_days = t_year*365 #DAYS

k = 0.5 #diffusivity in m^2/day 
u = 0.05 #advection velocity in m/day
c0 = 1 #mg/L

#For simplicity
D = (k*dT)/(dX**2)
S = (u*dT)/dX

#%% Part A of Qual Question

# Create empty matrix for c as a function of x for t - 1, 3.5, 5 years
c_ana = np.empty((len(t_year),len(x)))

for j in range(len(t_year)):
    for i in range(len(x)):
        c_ana[j,i] = (c0/2)*(math.erfc((x[i]-u*t_days[j])/(2*np.sqrt(k*t_days[j]))) + \
                     np.exp(u*x[i]/k)*math.erfc((x[i]+u*t_days[j])/(2*np.sqrt(k*t_days[j]))))

plt.figure()
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
for j in range(len(t_days)):
    color = ['#277AC4','#4F4A90', '#904A68'] 
    plt.plot(x,c_ana[j,:], color = color[j], label = 't = %.1f years, analytical' % t_year[j] )
    
plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
plt.savefig('C:/Users/Amanda/Desktop/PartA.eps', format='eps', dpi=1000)

#%% Part D of Qual Question

#Create A matrix
A = np.zeros((len(x),len(x)))

  
for j in range(len(x)-1):    
    A[j+1, j] = D+S/2
    A[j, j+1] = D-S/2
    
for i in range(len(x)):
    A[i,i] = 1-2*D 
    
A[n,n] = 1-2*D    

#Create B matrix
g1 = c0
g2 = 0

B = np.zeros(len(x))

B[0] = g1*(D+S/2)
B[n] = g2*(D-S/2)

#Solve equation
plt.figure()

for l in range(len(T)):
    
    c = np.zeros((T[l], n+1))

    for g in range(T[l]-1):
        c[g,0] = 1
        c[g+1, :] = np.matmul(A, c[g,:]) + B
    
    color = ['#277AC4','#4F4A90', '#904A68']
    plt.plot(x, c[g, :], linestyle = '-.', color = color[l], label = 't = %.1f years, numerical' % t_year[l])    


for j in range(len(t_days)):   
    color = ['#277AC4','#4F4A90', '#904A68'] 
    plt.plot(x,c_ana[j,:], color = color[j], label = 't = %.1f years, analytical' % t_year[j] )

plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
plt.savefig('C:/Users/Amanda/Desktop/PartD.eps', format='eps', dpi=1000)
    
#%% Part E of Qual Question
Pe = (u*dX)/k

k_new = k/10**3

Pe_new = (u*dX)/k_new

D_new = (k_new*dT)/(dX**2)
S_new = (u*dT)/dX


#Create new A matrix
A_new = np.zeros((len(x),len(x)))

  
for j in range(len(x)-1):    
    A_new[j+1, j] = D_new+S_new/2
    A_new[j, j+1] = D_new-S_new/2
    
for i in range(len(x)):
    A_new[i,i] = 1-2*D_new 
    
A_new[n,n] = 1-2*D_new    

#Create new B matrix
g1 = c0
g2 = 0

B_new = np.zeros(len(x))

B_new[0] = g1*(D+S/2)
B_new[n] = g2*(D-S/2)

#Solve equation
for l in range(len(T)):
    
    c_new = np.zeros((T[l], n+1))

    for g in range(T[l]-1):
        c_new[g, 0] = 1 #initial condition
        c_new[g+1, :] = np.matmul(A_new, c_new[g,:]) + B_new
    

plt.figure()
color = ['#277AC4']
plt.plot(x, c_new[T[0]-1, :], linestyle = '-.', color = color[0], label = 't = %.1f years, numerical, with $\kappa$ = 0.0005' % t_year[0])    


for j in range(len(t_days)):   
    color = ['#277AC4','#4F4A90', '#904A68'] 
    plt.plot(x,c_ana[j,:], color = color[j], label = 't = %.1f years, analytical' % t_year[j] )

plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
plt.savefig('C:/Users/Amanda/Desktop/PartE.eps', format='eps', dpi=1000)

#%% Part G of Qual Question

plt.figure()
R = [1, 1, 1, 1, 0.5, 1.2, 1.5]
gam = [0, 0.01, 0.1, 1, 0, 0, 0, 0]

for h in range(len(R)):
    Pe = (u*dX)/k
    W = (gam[h]*dT)/R[h]
    Z = (k*dT)/(R[h]*(dX)**2)

    #Create new A matrix
    A_all = np.zeros((len(x),len(x)))
    
    for j in range(len(x)-1):    
        A_all[j+1, j] = (1+Pe/2)*Z
        A_all[j, j+1] = (1-Pe/2)*Z
        
    for i in range(len(x)):
        A_all[i,i] = 1 - (2*Z) - W 
        
    A_all[n,n] = 1 - (2*Z) - W    
    
    #Create new B matrix
    g1 = c0
    g2 = 0
    
    B_all = np.zeros(len(x))
    
    B_all[0] = g1*((1+Pe/2)*Z)
    B_all[n] = g2*((1-Pe/2)*Z)
    
    #Solve equation
    for l in range(len(T)):
        
        c_all = np.zeros((T[l], n+1))
        
        for g in range(T[l]-1):
            c_all[g, 0] = 1 #initial condition
            c_all[g+1, :] = np.matmul(A_all, c_all[g,:]) + B_all

   
    color = ['#277AC4', '#904F4A', '#8D904A', '#90704A', '#56904A', '#514A90', '#4A8490']
    plt.plot(x, c_all[T[0]-1, :], linestyle = '-.', color = color[h], label = 'R = %.2f, ' % R[h] + '$\gamma$ = %.2f' % gam[h])    
plt.plot(x,c_ana[0,:], color = color[0], label = 't = %.1f year, analytical' % t_year[0])

plt.xlabel('X (m)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
plt.savefig('C:/Users/Amanda/Desktop/PartG.eps', format='eps', dpi=1000)