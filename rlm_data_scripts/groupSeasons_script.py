# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon

df = pd.read_csv('./rlm_output/groupedStorms.csv', index_col='date')
df.index = pd.to_datetime(df.index)

df['intensity_mm'] = df['intensity']*25.4 #Fix this. What is this intensity?

noZeros = df.loc[(df['stormNo']>=0) & (df['intensity_mm']>0)]
Zeros = df.loc[(df['stormNo']>=0) & (df['intensity_mm']<=0)]

noZeros = noZeros.copy()
Zeros = Zeros.copy()

noZeros['month'] = pd.to_datetime(noZeros.index).month
noZeros.loc[(noZeros['month'] == 12) | 
            (noZeros['month'] == 1) | 
            (noZeros['month'] ==2), 'season'] = 'Winter'
noZeros.loc[(noZeros['month'] == 3) | 
            (noZeros['month'] == 4) | 
            (noZeros['month'] ==5), 'season'] = 'Spring'
noZeros.loc[(noZeros['month'] == 6) | 
            (noZeros['month'] == 7) | 
            (noZeros['month'] ==8), 'season'] = 'Summer'
noZeros.loc[(noZeros['month'] == 9) | 
            (noZeros['month'] == 10) | 
            (noZeros['month'] ==11), 'season'] = 'Fall'

noZeros['seasonCount'] = noZeros.groupby('season')['season'].transform('count')

noZeros['mean_val'] = noZeros.groupby('season')['intensity_mm'].transform('mean')

plt.close('all')
fig, ax = plt.subplots(2,2)
winterData = noZeros.intensity_mm[noZeros.season=='Winter']
winterHist, winterEdges = np.histogram(winterData, bins=500)
normWinterHist = winterHist/len(winterData)
ax[0,0].bar(winterEdges[:-1], normWinterHist, align='edge', 
    edgecolor='b', width=0.075)
ax[0,0].set_title('Winter')

springData = noZeros.intensity_mm[noZeros.season=='Spring']
springHist, springEdges = np.histogram(springData, bins=500)
normSpringHist = springHist/len(springData)
ax[0,1].bar(springEdges[:-1], normSpringHist, align='edge', 
    color='pink', edgecolor='m', width=0.075)
ax[0,1].set_title('Spring')

summerData = noZeros.intensity_mm[noZeros.season=='Summer']
summerHist, summerEdges = np.histogram(summerData, bins=500)
normSummerHist = summerHist/len(summerData)
ax[1,0].bar(summerEdges[:-1], normSummerHist, align='edge', 
    color='lightgreen', edgecolor='g', width=0.075)
ax[1,0].set_title('Summer')

fallData = noZeros.intensity_mm[noZeros.season=='Fall']
fallHist, fallEdges = np.histogram(fallData, bins=500)
normFallHist = fallHist/len(fallData)
ax[1,1].bar(fallEdges[:-1], normFallHist, align='edge', 
    color='lightblue', edgecolor='c', width=0.075)
ax[1,1].set_title('Fall')

for ax in ax.flat:
    ax.set(xlabel='Rainfall intensity (mm/hr)', ylabel='Probability')
    ax.set(xlim=(0,20), ylim=(0,1.0))

plt.tight_layout()
plt.show()

# fig1, ax1 = plt.subplots()
# x = np.linspace(0, 13, 90)
# lambd = 1/noZeros.mean_val[noZeros.season == 'Winter']
# y = lambd[0]*np.exp(-lambd[0]*x)
# plt.plot(x,y)
# plt.bar(winterEdges[:-1], normWinterHist, align='edge', edgecolor='b', width=0.075)
# plt.show()