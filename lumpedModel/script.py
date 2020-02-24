# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime

xticks = pd.date_range(datetime.datetime(1981,1,1), datetime.datetime(2016,1,1), freq='5YS')

test = pd.read_pickle(r"C:\Users\Amanda\Documents\volcanic_data.pkl")

pnnl_precip = test['PREC_ACC_NC_hourlywrf_pnnl']

fig, ax = plt.subplots(figsize=(9,4))
pnnl_precip.iloc[:,0].plot(ax=ax, color ='navy', linewidth =0.75, xticks=xticks.to_pydatetime())

ax.tick_params('x', length=5, which='major')
ax.tick_params('x', length=2, which='minor')
ax.tick_params('both', bottom=True, top=True, left=True, right=True, which='both')
ax.set_xticklabels([x.strftime('%Y') for x in xticks])

ax.set_xlim(pd.Timestamp('1981'), pd.Timestamp('2016'))
plt.ylim(-0.5, 18.0)
plt.xlabel('Year')
plt.ylabel('Rainfall depth (mm)')

plt.text(0.6875, 0.925 , r'Location = (46.162$\degree$N, 122.61$\degree$W)',\
         bbox=dict(facecolor='white', edgecolor='lightgray'), transform=ax.transAxes)
plt.tight_layout()
plt.savefig(r'C:\Users\Amanda\Desktop\ESS519_Figure.svg')