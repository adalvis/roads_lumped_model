"""
Author: Amanda Manaster
Date: 20180506
Purpose: Functions called in HW2_Script.py.

"""
#%% Import Python libraries

import numpy as np

#%% Forward Euler
  
def fwd(f, t_s, t_e, n, u_ini):
    dT = (t_e-t_s)/n
    u_fwd = np.zeros(n)
    t = np.linspace(t_s, t_e, num = n)
    u_fwd[0] = u_ini
    
    for i in range(n-1):
        u_fwd[i+1] = f(u_fwd[i],t[i])*dT+ u_fwd[i]

        
    return(t, u_fwd)    
    
#%% Backward Euler

    
def bwd(f, df, t_s, t_e, n, u_ini, tol):
    dT = (t_e-t_s)/n
    u_bwd = np.zeros(n)
    t = np.linspace(t_s, t_e, num = n)
    u_bwd[0] = u_ini
    t_old = t_s
    u_old = u_ini 
    
    
    for i in range(n-1):
        
        t_new = t_old + dT
        
        dsol = 0.1
        
        while dsol > tol:
            u_new = u_old - (f(u_old, t_old)*dT + u_bwd[i] - u_old)/(df(u_old, t_old)*dT-1)
            dsol = np.abs(u_new-u_old)
            u_old = u_new
            
        t_old = t_new
        u_bwd[i+1] = u_new
        t[i+1] = t_new
        
    return(t, u_bwd)     
    

#%% Second-Order Runge Kutta

def rk2(f, t_s, t_e, n, u_ini):
    dT = (t_e-t_s)/n
    u_cor = np.zeros(n)
    u_star = np.zeros(n)
    t = np.linspace(t_s, t_e, num = n)
    u_cor[0] = u_ini
    
    for i in range(n-1):
        u_star[i] = u_cor[i]+dT*(f(u_cor[i],t[i]))
        u_cor[i+1] = u_cor[i] + (dT/2)*(f(u_cor[i],t[i]) + f(u_star[i], t[i]+dT))

        
    return(t, u_cor)    
    
#%% Fourth-order Runge Kutta

def rk4(f, t_s, t_e, n, u_ini):
    dT = (t_e-t_s)/n
    u_4 = np.zeros(n)
    u_star = np.zeros(n)
    u_star2 = np.zeros(n)
    u_star3 = np.zeros(n)
    t = np.linspace(t_s, t_e, num = n)
    u_4[0] = u_ini
    
    for i in range(n-1):
        u_star[i] = u_4[i]+(dT/2)*(f(u_4[i],t[i]))
        u_star2[i] = u_4[i]+(dT/2)*(f(u_star[i],t[i]+(dT/2)))
        u_star3[i] = u_4[i]+(dT)*(f(u_star2[i],t[i]+(dT/2)))
        u_4[i+1] = u_4[i] + (dT/6)*(f(u_4[i],t[i]) + 2*f(u_star[i], t[i]+(dT/2)) + 2*f(u_star2[i],t[i]+(dT/2)) + f(u_star3[i],t[i]+(dT)))
        
        
    return(t, u_4) 