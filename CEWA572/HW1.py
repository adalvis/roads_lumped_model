# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:48:31 2018

@author: Amanda
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'bold'

x = np.arange(-20, 20, 1)
t = np.arange(-20, 20, 1)
y = np.zeros((40,40))

plt.figure()
for xc in x:
    plt.axvline(x = xc, color = 'b', linestyle = '-')

for j in range(40):
    for i in range(40):
        y[i,j] = (1/3)*(x[j]-x[i])+t[i]
        
for l in range(40):
    plt.plot(x, y[l], 'r--')        

ax = plt.gca()
ax.tick_params(axis='both', which='both', direction = 'in', bottom = 'on', 
               left = 'on', top = 'on', right = 'on')
plt.xlabel('x', fontweight = 'bold')
plt.ylabel('t', fontweight = 'bold')
plt.xlim(0,8)
plt.ylim(0,8)
plt.xticks([0,8],('x = 0', 'x = L'))
plt.yticks([0,8],('t = 0', 't = T'))
#plt.savefig('C://Users/Amanda/Desktop/HW1_Q2.png', bbox_inches = 'tight')
plt.show()
