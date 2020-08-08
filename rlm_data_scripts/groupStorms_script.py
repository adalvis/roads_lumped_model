# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon


data = pd.read_csv('./rlm_output/ElkRock_rain_10yr.csv', index_col='date') #for 6 yr
# data = pd.read_csv('./rlm_output/ElkRock_rain_14yr.csv', index_col='date') #for 14 yr
# data = pd.read_csv('./rlm_output/ElkRock_rain_10yr.csv', index_col='date') #for 10 yr
#data = pd.read_csv('./rlm_output/ElkRock_rain.csv', index_col='date') #for 1 yr
data.index = pd.to_datetime(data.index)
data = data[data.index >= data.index[12614]] #for 7 yr
data = data.asfreq('H')
data[data['intensity_mmhr']<0]=0
data.fillna(0, inplace=True)
df = data.copy()

fig1, ax1 = plt.subplots()
df.intensity_mmhr.plot(ax=ax1, color='navy', linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Intensity (mm/hr)')
plt.title('Raw Elk Rock data')
plt.tight_layout()
#plt.savefig(r'C:/Users/Amanda/Documents/GitHub/roads_lumped_model/rlm_output/rawData_ElkRock.png')
plt.show()

time_since_rain = np.zeros(len(df))

if df.intensity_mmhr[0]==0:
    time_since_rain[0] = 1

for (i, entry) in enumerate(df.intensity_mmhr):
    if entry == 0 and i != 0:
        time_since_rain[i] = time_since_rain[i-1] + 1
 
storm_index = np.empty(len(df))
storm_index[:] = None
storm_no = 0
rainStart = 0 
rainEnd = 0
Tb_flag = False

total = np.zeros(len(df))

# Loop through time since rain. j gives index, val gives the value at j
for (j, val) in enumerate(time_since_rain):
	# If val == 0 (i.e., there is rain at this time step) and Tb_flag has been set
	# to true (i.e., a 3 has been spotted), give storm_index the value of
	# storm_no for a range of rainStart to rainEnd+1 (because ranges aren't inclusive
	# in list slicing). Then, increase storm_no by one, restart rainStart at j and
	# rainEnd at j, and switch Tb_flag back to false
	if (val == 0 and Tb_flag):
		storm_index[rainStart:rainEnd+1] = storm_no
		storm_no += 1
		rainStart = j
		rainEnd = j
		Tb_flag = False
	# If val == 0 (i.e., there is rain at this time step), make rainEnd equal to
	# this index
	elif val == 0:
		rainEnd = j
	# If val == 3, set the Tb_flag to True (i.e., there has been enough time without
	# rain for it to begin another storm
	elif val == 3:
		Tb_flag = True
	
	# For the end of the list, if the index is of the last value in the list,
	# make sure that rainEnd is equal to this index, and assign storm_index the
	# the value of storm_no for range of rainStart to rainEnd+1
	if (j == len(time_since_rain)-1):
		rainEnd = j
		storm_index[rainStart:rainEnd+1] = storm_no

df['timeSinceRain'] = time_since_rain
df['stormNo'] = storm_index
df['totalNo'] = df.stormNo.copy()
df.totalNo.fillna(method='ffill', inplace=True)
df['groupedDepth'] = df.groupby('stormNo')['intensity_mmhr'].transform('sum')
df.stormNo.fillna(-0.01, inplace=True)
df.groupedDepth.fillna(0.0, inplace = True)

fig2, ax2 = plt.subplots()
df.groupedDepth.plot(ax=ax2, color='teal', linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Rainfall depth (mm)')
plt.title('Total storm depth Elk Rock data')
plt.tight_layout()
#plt.savefig(r'C:/Users/Amanda/Documents/GitHub/roads_lumped_model/rlm_output/groupedData_ElkRock.png')
plt.show()

# Save output
df.to_csv('./rlm_output/groupedStorms_ElkRock_7yr.csv') #for 7 yr
#df.to_csv('./rlm_output/groupedStorms_ElkRock_14yr.csv') #for 14 yr
#df.to_csv('./rlm_output/groupedStorms_ElkRock_10yr.csv') #for 10 yr
#df.to_csv('./rlm_output/groupedStorms_ElkRock.csv') #for 1 yr

#Below is code I used to hack my way through this the first time. The cleaner
#version of this is above!

# storm_index_2 = np.empty(len(df))
# storm_index_2[:] = -9999
# storm_no = 0
# for (j, val) in enumerate(time_since_rain):
#     if j >= (len(time_since_rain)-12):
#     	storm_index_2[j] = storm_no
#     	total[j] = storm_no
#     elif val == 0:
#         storm_index_2[j] = storm_no
#         total[j] = storm_no
#     elif val == 12:
#         storm_no += 1
#         total[j] = storm_no - 1
#     elif val == 1:
#     	storm_index_2[j] = storm_no if time_since_rain[j+11] != 12 else -9999
#     	total[j] = storm_no
#     elif val == 2:
#         storm_index_2[j] = storm_no if time_since_rain[j+10] != 12 else -9999
#         total[j] = storm_no
#     elif val == 3:
#         storm_index_2[j] = storm_no if time_since_rain[j+9] != 12 else -9999
#         total[j] = storm_no
#     elif val == 4:
#         storm_index_2[j] = storm_no if time_since_rain[j+8] != 12 else -9999
#         total[j] = storm_no 
#     elif val == 5:
#         storm_index_2[j] = storm_no if time_since_rain[j+7] != 12 else -9999
#         total[j] = storm_no
#     elif val == 6:
#         storm_index_2[j] = storm_no if time_since_rain[j+6] != 12 else -9999
#         total[j] = storm_no
#     elif val == 7:
#         storm_index_2[j] = storm_no if time_since_rain[j+5] != 12 else -9999
#         total[j] = storm_no
#     elif val == 8:
#         storm_index_2[j] = storm_no if time_since_rain[j+4] != 12 else -9999
#         total[j] = storm_no 
#     elif val == 9:
#         storm_index_2[j] = storm_no if time_since_rain[j+3] != 12 else -9999
#         total[j] = storm_no
#     elif val == 10:
#         storm_index_2[j] = storm_no if time_since_rain[j+2] != 12 else -9999
#         total[j] = storm_no
#     elif val == 11:
#         storm_index_2[j] = storm_no if time_since_rain[j+1] != 12 else -9999
#         total[j] = storm_no
#     else:
#         storm_index_2[j] = -9999
#         total[j] = storm_no-1 if time_since_rain[j] >= 12 else storm_no