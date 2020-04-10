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


#Calculate mean and standard deviation for storms WITHOUT ZEROS
noZeros['mean_val'] = noZeros.groupby('stormNo')['stationOne'].transform('mean')
noZeros['stdev_val'] = noZeros.groupby('stormNo')['stationOne'].transform('std')

#%% Playing w/ cumulative histograms, etc
Rnondim = np.linspace(expon.ppf(0.01, scale=noZeros.mean_val[noZeros.stormNo == 7.0]), 
    expon.ppf(0.985, scale=noZeros.mean_val[noZeros.stormNo == 7.0]), 10) #nondimensional x axis
CF = expon.cdf(Rnondim, scale=noZeros.mean_val[noZeros.stormNo == 7.0]) #cumulative probability of flows

fig5, ax5 = plt.subplots()
noZeros.stationOne[noZeros.stormNo == 7.0].hist(bins=30, density=True, 
                                                   cumulative = True, ax=ax5)
plt.plot(Rnondim,CF, '-')
plt.xlabel('Rainfall (mm/hr)')
plt.ylabel('Cumulative frequency')
plt.title('Highest instantaneous intensity storm histogram')
#plt.show()

# #%% Brute force PDF for one storm
fig6, ax6 = plt.subplots()
noZeros.stationOne[noZeros.stormNo == 7.0].hist(bins=10, density=True, ax=ax6)
#plt.show()

fig7, ax7 = plt.subplots()
x = np.linspace(0, 9, 90)
x1 = np.linspace(expon.ppf(0.01, scale=noZeros.mean_val[noZeros.stormNo == 7.0]), 
    expon.ppf(0.9999, scale=noZeros.mean_val[noZeros.stormNo == 7.0]), 10)
lambd = 1/noZeros.mean_val[noZeros.stormNo == 7.0]
y = lambd[0]*np.exp(-lambd[0]*x)
y1 = expon.pdf(x1, scale=noZeros.mean_val[noZeros.stormNo == 7.0])
noZeros.stationOne[noZeros.stormNo == 7.0].hist(bins=10, density=True, ax=ax7)
plt.plot(x, y)
plt.plot(x1,y1)
plt.show()

#%%Trying to do the analysis Erkan did?
# Rnondim = np.linspace(0, 20, 30) #nondimensional x axis
# CF = expon.cdf(Rnondim, scale = 0.75) #cumulative probability of flows
# plt.figure()
# plt.plot(Rnondim,CF,'-')
# plt.xlabel('Rainfall_n (-)') 
# plt.ylabel('Cumulative Probability')
# plt.show()

# Df = np.zeros(len(Rnondim))
# Df[0] = CF[0]

# for x in range(1, len(Rnondim)):
#     Df[x] = -CF[x-1]+CF[x]

# FDur = np.multiply(np.array([Df]).T, timeStep) #flow duration

# plt.figure()
# plt.plot(Rnondim,FDur,'-')
# #plt.hist(data, bins = 40)
# plt.xlabel('Rainfall_n (-)') 
# plt.ylabel('Storm Duration (hours)')
