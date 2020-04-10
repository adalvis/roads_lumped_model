# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon

data = pd.read_pickle('C:/Users/Amanda/Documents/volcanic_data.pkl')
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
noZeros['mean'] = noZeros.groupby('stormNo')['stationOne'].transform('mean')
noZeros['stdev'] = noZeros.groupby('stormNo')['stationOne'].transform('std')

#%% Longest storm = 564 hrs; high intensity = 6.84 mm/hr
fig1, ax1 = plt.subplots()
noZeros.stationOne[noZeros.stormDuration == noZeros.stormDuration.max()].plot()
plt.title('Longest storm hyetograph')
plt.show()

fig2, ax2 = plt.subplots()
counts, bins = np.histogram(noZeros.stationOne[noZeros.stormDuration == noZeros.stormDuration.max()], 
                            density = True, bins = 30)
plt.hist(bins[:-1], bins, weights=counts)
plt.xlabel('Rainfall (mm/hr)')
plt.ylabel('Frequency')
plt.title('Longest storm histogram')
plt.show()

#%% Highest intensity = 17 mm/hr; storm duration = 84 hr
fig3, ax3 = plt.subplots()
noZeros.stationOne[noZeros.stormNo == 1705.0].plot()
plt.title('Highest instantaneous intensity storm hyetograph')
plt.show()

fig4, ax4 = plt.subplots()
counts1, bins1 = np.histogram(noZeros.stationOne[noZeros.stormNo == 1705.0], 
                            density = True, bins = 30)
plt.hist(bins1[:-1], bins1, weights=counts1)
plt.xlabel('Rainfall (mm/hr)')
plt.ylabel('Frequency')
plt.title('Highest instantaneous intensity storm histogram')
plt.show()

#%% Playing w/ cumulative histograms, etc
Rnondim = np.linspace(0, 17, 30) #nondimensional x axis
CF = expon.cdf(Rnondim, scale = 1) #cumulative probability of flows

fig5, ax5 = plt.subplots()
noZeros.stationOne[noZeros.stormNo == 1705.0].hist(bins=30, density=True, 
                                                   cumulative = True, ax=ax5)
plt.plot(Rnondim,CF, '-')
plt.xlabel('Rainfall (mm/hr)')
plt.ylabel('Cumulative frequency')
plt.title('Highest instantaneous intensity storm histogram')
plt.show()

#%% Brute force PDF for one storm
data = np.array(noZeros.stationOne[noZeros.stormNo == 1705.0])
fig6, ax6 = plt.subplots()
hist, edges = np.histogram(data, bins = 30)
norm_hist = hist/len(data)
plt.bar(edges[:-1], norm_hist, align='edge')
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


#%% Storm time step
# fig1, ax1 = plt.subplots(figsize=(9,4))
# xticks1 = pd.date_range(datetime.datetime(1981,1,1), datetime.datetime(2016,1,1), freq='5YS')
# df.stormDepth.plot(ax=ax1, color='magenta', linewidth=0.75, xticks=xticks1.to_pydatetime())

# ax1.tick_params('x', length=5, which='major')
# ax1.tick_params('x', length=2, which='minor')
# ax1.tick_params('both', bottom=True, top=True, left=True, right=True, which='both')
# ax1.set_xticklabels([x.strftime('%Y') for x in xticks1])

# ax1.set_xlim(pd.Timestamp('1981'), pd.Timestamp('2016'))
# plt.ylim(-0.5, 500)
# plt.xlabel('Year')
# plt.ylabel('Rainfall depth (mm)')
# plt.title('Per storm rainfall depth over 35 years', fontweight='bold')

# plt.text(0.6875, 0.925 , r'Location = (46.162$\degree$N, 122.61$\degree$W)',\
#           bbox=dict(facecolor='white', edgecolor='lightgray'), transform=ax1.transAxes)
# plt.tight_layout()
# plt.show()

#%% Hourly time step
# fig2, ax2 = plt.subplots(figsize=(9,4))
# xticks2 = pd.date_range(datetime.datetime(1981,1,1), datetime.datetime(2016,1,1), freq='5YS')
# staOne.iloc[:].plot(ax=ax2, color ='navy', linewidth =0.75, xticks=xticks2.to_pydatetime())

# ax2.tick_params('x', length=5, which='major')
# ax2.tick_params('x', length=2, which='minor')
# ax2.tick_params('both', bottom=True, top=True, left=True, right=True, which='both')
# ax2.set_xticklabels([x.strftime('%Y') for x in xticks2])

# ax2.set_xlim(pd.Timestamp('1981'), pd.Timestamp('2016'))
# plt.ylim(-0.5, 18.0)
# plt.xlabel('Year')
# plt.ylabel('Rainfall depth (mm)')
# plt.title('Per hour rainfall depth over 35 years', fontweight='bold')

# plt.text(0.6875, 0.925 , r'Location = (46.162$\degree$N, 122.61$\degree$W)',\
#           bbox=dict(facecolor='white', edgecolor='lightgray'), transform=ax2.transAxes)
# plt.tight_layout()
# # #plt.savefig(r'C:\Users\Amanda\Desktop\ESS519_Figure.svg')
# plt.show()