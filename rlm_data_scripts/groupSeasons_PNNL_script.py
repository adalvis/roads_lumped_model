# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon

data = pd.read_pickle('/mnt/c/Users/Amanda/Documents/volcanic_data.pkl')
pnnl_precip = data['PREC_ACC_NC_hourlywrf_pnnl']

df = pnnl_precip.iloc[:,0].to_frame(name = 'stationOne')
no_rain = df.iloc[4,:]
t0 = df.index[0]
deltaT = datetime.timedelta(hours=1)

hours_since_rain = np.zeros(len(df))

for (i, entry) in enumerate(df.values):
    if entry == 0 and i != 0:
        hours_since_rain[i] = hours_since_rain[i-1] + 1
 
storm_index = np.empty(len(df))
storm_index[:] = None
storm_no = 0
total = np.zeros(len(df))

for (j, val) in enumerate(hours_since_rain):
    if val == 0:
        storm_index[j] = storm_no
        total[j] = storm_no
    elif val == 3:
        storm_no += 1
        total[j] = storm_no - 1
    elif val == 1:
        storm_index[j] = storm_no if hours_since_rain[j+2] != 3 else None
        total[j] = storm_no
    elif val == 2:
        storm_index[j] = storm_no if hours_since_rain[j+1] != 3 else None
        total[j] = storm_no
    else:
        storm_index[j] = None
        total[j] = storm_no-1 if hours_since_rain[j] >= 3 else storm_no

df['stormNo'] = storm_index
df['totalNo'] = total
df['stormDepth'] = df.groupby('stormNo')['stationOne'].transform('sum')
df['stormDuration'] = df.groupby('stormNo')['stationOne'].transform('count')
df['stormIntensity'] = df.stormDepth/df.stormDuration
df['timeStep'] = df.groupby('totalNo')['totalNo'].transform('count')

df.fillna(-9999, inplace=True)

noZeros = df.loc[(df['stormNo']>=0) & (df['stationOne']>0)]
Zeros = df.loc[(df['stormNo']>=0) & (df['stationOne']<=0)]

noZeros = noZeros.copy()
Zeros = Zeros.copy()

noZeros['month'] = pd.to_datetime(noZeros.index).month
noZeros.loc[(noZeros['month'] == 12) | (noZeros['month'] == 1) | (noZeros['month'] ==2), 'season'] = 'Winter'
noZeros.loc[(noZeros['month'] == 3) | (noZeros['month'] == 4) | (noZeros['month'] ==5), 'season'] = 'Spring'
noZeros.loc[(noZeros['month'] == 6) | (noZeros['month'] == 7) | (noZeros['month'] ==8), 'season'] = 'Summer'
noZeros.loc[(noZeros['month'] == 9) | (noZeros['month'] == 10) | (noZeros['month'] ==11), 'season'] = 'Fall'

noZeros['seasonCount'] = noZeros.groupby('season')['season'].count()

noZeros['mean_val'] = noZeros.groupby('season')['stationOne'].transform('mean')

fig, ax = plt.subplots(2,2)
winterData = noZeros.stationOne[noZeros.season=='Winter']
winterHist, winterEdges = np.histogram(winterData, bins=100)
normWinterHist = winterHist/len(winterData)
ax[0,0].bar(winterEdges[:-1], normWinterHist, align='edge', edgecolor='b', width=0.075)
ax[0,0].set_title('Winter')

springData = noZeros.stationOne[noZeros.season=='Spring']
springHist, springEdges = np.histogram(springData, bins=100)
normSpringHist = springHist/len(springData)
ax[0,1].bar(springEdges[:-1], normSpringHist, align='edge', 
    color='m', edgecolor='m', width=0.075)
ax[0,1].set_title('Spring')

summerData = noZeros.stationOne[noZeros.season=='Summer']
summerHist, summerEdges = np.histogram(summerData, bins=100)
normSummerHist = summerHist/len(summerData)
ax[1,0].bar(summerEdges[:-1], normSummerHist, align='edge', 
    color='g', edgecolor='g', width=0.075)
ax[1,0].set_title('Summer')

fallData = noZeros.stationOne[noZeros.season=='Fall']
fallHist, fallEdges = np.histogram(fallData, bins=100)
normFallHist = fallHist/len(fallData)
ax[1,1].bar(fallEdges[:-1], normFallHist, align='edge', 
    color='c', edgecolor='c', width=0.075)
ax[1,1].set_title('Fall')

for ax in ax.flat:
    ax.set(xlabel='Rainfall rate (mm/hr)', ylabel='Probability')

plt.tight_layout()
plt.show()

fig1, ax1 = plt.subplots()
x = np.linspace(0, 13, 90)
lambd = 1/noZeros.mean_val[noZeros.season == 'Winter']
y = lambd[0]*np.exp(-lambd[0]*x)
plt.plot(x,y)
plt.bar(winterEdges[:-1], normWinterHist, align='edge', edgecolor='b', width=0.075)
plt.show()