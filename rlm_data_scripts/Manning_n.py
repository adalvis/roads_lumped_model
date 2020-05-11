"""
Author: Amanda Manaster
Date: 05/11/2020
Purpose: To show relationship between Manning's n, q, f_s, tau, H
"""

import numpy as np
import matplotlib.pyplot as plt

S_f = np.arange(0, 0.0625, 0.00001)
q = np.linspace(7.43e-6, 0.0003855, len(S_f))
d95 = 0.0275
rho = 1000
g = 9.81
S = 0.03

n_f = 0.0026*q**-0.274
n_c = 0.08*q**-0.153

n_t = np.zeros((len(q),len(q)))

for i, val in enumerate(S_f):
	if val <= d95:
		n_t[:,i] = n_c + val/d95*(n_f-n_c)
	else:
		n_t[:,i] = n_f

