# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon

data = pd.read_csv('/home/adalvis/github/roads_lumped_model/'+\
	'rlm_data/ElkRock_rain_10yr.csv', index_col='date')

data.index = pd.to_datetime(data.index)
data = data.asfreq('h')
data[data['intensity_mmhr']<0]=0
data.fillna(0, inplace=True)
df = data.copy()

fig1, ax1 = plt.subplots()
df.intensity_mmhr.plot(ax=ax1, color='navy', linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Intensity (mm/hr)')
plt.title('Raw Elk Rock data')
plt.tight_layout()
plt.show()

df['dates'] = df.index.date
df['daily_mean'] = df.groupby('dates')['intensity_mmhr'].transform('mean')
df['daily_depth'] = df.groupby('dates')['intensity_mmhr'].transform('sum')

fig2, ax2 = plt.subplots()
df.daily_depth.plot(ax=ax2, color='teal', linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Rainfall depth (mm)')
plt.title('Total daily depth Elk Rock data')
plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots()
df.daily_mean.plot(ax=ax3, color='mediumvioletred', linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Rainfall depth (mm)')
plt.title('Mean daily depth Elk Rock data')
plt.tight_layout()
plt.show()

# Save output
# df.to_csv('/home/adalvis/github/roads_lumped_model/'+\
# 	'rlm_data/dailyStorms_ElkRock10.csv')