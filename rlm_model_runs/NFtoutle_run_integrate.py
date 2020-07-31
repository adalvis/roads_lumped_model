#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import NFtoutle_module_integrate as mod

import importlib
importlib.reload(mod)

data_df = pd.read_csv('./rlm_output/groupedStorms_ElkRock_10yr.csv', index_col='date')
data_df.index = pd.to_datetime(data_df.index)

k_cs = [0.25e-7, 0.25e-7, 0.25e-7, 0.25e-7, 0.25e-7, 
        0.5e-7, 0.5e-7, 0.5e-7, 0.5e-7, 0.5e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        2e-7, 2e-7, 2e-7, 2e-7, 2e-7, 
        4e-7, 4e-7, 4e-7, 4e-7, 4e-7]
k_cb = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
q_ps = [0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7]
q_pb = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
e = 0.725

sed = np.zeros(len(k_cs))
storms_df = []

for i in range(len(k_cs)):
    sed[i], storms_df = mod.model_run(data_df, k_cs[i], k_cb[i], q_ps[i], q_pb[i], e)

sed = sed/10
q_rat = [i / j for i, j in zip(q_ps, q_pb)]
k_rat = [i / j for i, j in zip(k_cs, k_cb)]

plt.plot(q_rat[0:5],sed[0:5], 'o', label='1/4')
plt.plot(q_rat[5:10],sed[5:10], '*', label = '1/2')
plt.plot(q_rat[10:15],sed[10:15], '>', label = '1')
plt.plot(q_rat[15:20],sed[15:20], '^', label = '2')
plt.plot(q_rat[20:25],sed[20:25], 's', label = '4')
plt.xlabel('q ratio')
plt.ylabel('10-year average sediment yield (kg/m)')
plt.legend(title='k ratio')
plt.show()
