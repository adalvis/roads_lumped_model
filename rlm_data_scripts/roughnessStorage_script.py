import numpy as np
import matplotlib.pyplot as plt

S_f = np.arange(0, 0.05, 4.69e-6)
d95 = 0.0275
d50 = 6.25e-5
n_f = np.full((len(S_f)), 0.0475*d50**(1/6)) 
n_c = np.array([0.0475*(d95-S_f[i])**(1/6) if d95 >= S_f[i] 
	else 0 for i in range(len(S_f))])
n_t = np.array([n_f[i]+n_c[i] if d95 > S_f[i] else n_f[i] for i in range(len(S_f))])
f_s = (n_f/n_t)**1.5

plt.close('all')
fig1, ax1 = plt.subplots()
ax1.plot(S_f, n_c)
ax1.set_xlabel(r'$S_f$ [m]')
ax1.set_ylabel(r'$n_c$')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(S_f, n_t)
ax2.set_xlabel(r'$S_f$ [m]')
ax2.set_ylabel(r'$n_t$')
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(S_f, f_s)
ax3.set_xlabel(r'$S_f$ [m]')
ax3.set_ylabel(r'$f_s$')
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(n_t, f_s)
ax4.set_xlabel(r'$n_t$ [m]')
ax4.set_ylabel(r'$f_s$')
plt.show()