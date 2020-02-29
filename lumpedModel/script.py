# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


minTb = 3 #hours; threshold for determining an interstorm time period

data = pd.read_pickle(r"C:\Users\Amanda\Documents\volcanic_data.pkl")
pnnl_precip = data['PREC_ACC_NC_hourlywrf_pnnl']

staOne = pnnl_precip.iloc[:,0] #Get data for one station only

df = staOne.to_frame(name = 'Station_1')
no_rain = df.iloc[4,:]
t0 = df.index[0]
deltaT = datetime.timedelta(hours=1)

#%%
hours_since_rain = np.zeros(len(df))

for (i, entry) in enumerate(df.values):
    if entry == 0 and i != 0:
        hours_since_rain[i] = hours_since_rain[i-1] + 1
        
        
storm_index = np.empty(len(df))
storm_index[:] = np.nan
storm_no = 0

for (j, val) in enumerate(hours_since_rain):
    if val == 0:
        storm_index[j] = storm_no
    elif val == 3:
        storm_no +=1
        
# for (j, row) in enumerate(storm_index):
#     if storm_index[j] == np.nan and storm_index[j-1] != np.nan and storm_index[j+1] != np.nan:
#         storm_index[j] = storm_no
#     elif storm_index[j] == np.nan and storm_index[j-1] == np.nan and storm_index[j-2] != np.nan and storm_index[j+1] != np.nan:
#         storm_index[j] = storm_no
#     elif storm_index[j] == np.nan and storm_index[j+1] == np.nan and storm_index[j+2] != np.nan and storm_index[j-1] != np.nan:
#         storm_index[j] = storm_no
#%%
# fig, ax = plt.subplots(figsize=(9,4))
# xticks = pd.date_range(datetime.datetime(1981,1,1), datetime.datetime(2016,1,1), freq='5YS')
# staOne.iloc[:].plot(ax=ax, color ='navy', linewidth =0.75, xticks=xticks.to_pydatetime())

# ax.tick_params('x', length=5, which='major')
# ax.tick_params('x', length=2, which='minor')
# ax.tick_params('both', bottom=True, top=True, left=True, right=True, which='both')
# ax.set_xticklabels([x.strftime('%Y') for x in xticks])

# ax.set_xlim(pd.Timestamp('1981'), pd.Timestamp('2016'))
# plt.ylim(-0.5, 18.0)
# plt.xlabel('Year')
# plt.ylabel('Rainfall depth (mm)')

# plt.text(0.6875, 0.925 , r'Location = (46.162$\degree$N, 122.61$\degree$W)',\
#          bbox=dict(facecolor='white', edgecolor='lightgray'), transform=ax.transAxes)
# plt.tight_layout()
# #plt.savefig(r'C:\Users\Amanda\Desktop\ESS519_Figure.svg')