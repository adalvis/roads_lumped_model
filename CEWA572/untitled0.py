"""
Created on Wed Apr 25 13:48:38 2018

Author: Amanda
"""
import numpy as np

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