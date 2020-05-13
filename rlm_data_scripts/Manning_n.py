"""
Author: Amanda Manaster
Date: 05/11/2020
Purpose: To show relationship between Manning's n, q, f_s, tau, H
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl

S_f = np.arange(0, 0.0625, 0.00001)
q = np.linspace(7.43e-6, 0.0003855, len(S_f))
d95 = 0.0275
rho = 1000
g = 9.81
S = 0.03

n_f = 0.0026*q**-0.274
n_c = 0.08*q**-0.153

n_t = np.zeros((len(q),len(q)))
f_s = np.zeros((len(q),len(q)))
H = np.zeros((len(q),len(q)))

for i, val in enumerate(S_f):
	if val <= d95:
		n_t[:,i] = n_c + val/d95*(n_f-n_c)
	else:
		n_t[:,i] = n_f
	f_s[:,i] = (n_f/n_t[:,i])**1.5
	H[:,i] = (n_t[:,i]*q/S**(1/2))**(3/5)

tau = rho*g*H*S
tau_s = tau*f_s

plt.close('all')
fig, ax = plt.subplots(figsize=(8,5))
ax.tick_params(bottom=True, top=True, left=True, right=True, which='both')

norm = mpl.colors.Normalize(vmin=5e-6,vmax=0.0005)
cmap = plt.get_cmap('cividis_r', 25)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

for j, val in enumerate(np.arange(0, 6249, 250)):
	ax.plot(S_f, tau_s[val,:], c=cmap(j))

cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_ylabel(r'$q\ [m^2/s]$', fontsize=12)
ax.set_xlim(0.0, 0.07)
ax.set_ylim(0.0, 0.8)
ax.set_xlabel(r'Fine storage, $S_f$ [m]', fontsize=12)
ax.set_ylabel(r'$\tau_s$ [Pa]', fontsize=12)
ax.set_title(r'Grain shear as a function of $S_f$', fontsize=14)
plt.show()