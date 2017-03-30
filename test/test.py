#just testing spyder vs jupyter notebooks

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.stretch'] = 1
plt.figure(figsize = (7,6))


def unsubmerged(A, d, HW, K, M):
    Qu = A*d**0.5*(HW/(K*d))**(1/M)
    return Qu
    
def submerged(A, d, HW, Y, S, c):
    Qs = A*d**0.5*((HW/d-Y+0.5*S)/c)**(1/2)
    return Qs
    
HW = np.linspace(0, 12, 24)
b = 4
d = 5
A = b*d
S = 0.01
K = 0.486
M = 0.667
c = 0.0252
Y = 0.865

for i in range(24): 
        if HW[i] < 1.2*d:
            qu = unsubmerged(A, d, HW, K, M)
        elif HW[i] >= 1.2*d:
            qs = submerged(A, d, HW, Y, S, c)

            
plt.minorticks_on()
ax = plt.gca()
ax.tick_params(which = 'major', length = 5)
ax.tick_params(which = 'minor', length = 2)
plt.xlabel('Discharge (cfs)', fontsize = 14)
plt.ylabel('Headwater Elevation (ft)', fontsize = 14) 
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.plot(qu[0:12],HW[0:12])
plt.plot(qs[12:23],HW[12:23])
plt.show()