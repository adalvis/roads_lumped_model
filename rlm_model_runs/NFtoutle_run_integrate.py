#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import NFtoutle_module_integrate as mod

import importlib
importlib.reload(mod)

data_df = pd.read_csv('./rlm_output/groupedStorms_ElkRock_7yr.csv', index_col='date')
data_df.index = pd.to_datetime(data_df.index)


# k_cs = [0.25e-6, 0.25e-6, 0.25e-6, 0.25e-6, 0.25e-6, 
#         0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 
#         4e-6, 4e-6, 4e-6, 4e-6, 4e-6]
# k_cb = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
# u_ps = [0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6, 
#         0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6, 
#         0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6, 
#         0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6, 
#         0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6]
# u_pb = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
#         1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
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
u_ps = [0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7, 
        0.25e-7, 0.5e-7, 1e-7, 2e-7, 4e-7]
u_pb = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
e = 0.725

sed = np.zeros(len(k_cs))
storms_df = []

for i in range(len(k_cs)):
    sed[i], storms_df = mod.model_run(data_df, k_cs[i], k_cb[i], u_ps[i], u_pb[i], e)

sed = sed/7
u_rat = [i / j for i, j in zip(u_ps, u_pb)]
k_rat = [i / j for i, j in zip(k_cs, k_cb)]

plt.plot(u_rat[0:5],sed[0:5], 'o', label='1/4')
plt.plot(u_rat[5:10],sed[5:10], '*', label = '1/2')
plt.plot(u_rat[10:15],sed[10:15], '>', label = '1')
plt.plot(u_rat[15:20],sed[15:20], '^', label = '2')
plt.plot(u_rat[20:25],sed[20:25], 's', label = '4')
plt.xlabel('u ratio')
plt.ylabel('7-year average sediment yield (kg/m)')
plt.legend(title='k ratio')
plt.savefig(r'C:/Users/Amanda/Documents/GitHub/roads_lumped_model/rlm_output/7year_low_integrate.png')
plt.show()
