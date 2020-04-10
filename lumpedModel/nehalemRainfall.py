# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon

data = pd.read_csv('/mnt/c/Users/Amanda/Documents/nehalemFifteenMin.csv', index_col='date')

data.index = pd.to_datetime(data.index)


df = data.resample('15min').sum().fillna(0)



t0 = df.index[0]
deltaT = datetime.timedelta(minutes=15)

time_since_rain = np.zeros(len(df))

for (i, entry) in enumerate(df.depth):
    if entry == 0 and i != 0:
        time_since_rain[i] = time_since_rain[i-1] + 1
 
storm_index = np.empty(len(df))
storm_index[:] = None
storm_no = 0
total = np.zeros(len(df))

for (j, val) in enumerate(time_since_rain):
    if val >= 0 and val < 12:
        storm_index[j] = storm_no
        total[j] = storm_no
    elif val == 12:
        storm_no += 1
        total[j] = storm_no - 1
    # elif val == 1:
    #     storm_index[j] = storm_no if time_since_rain[j+2] != 3 else None
    #     total[j] = storm_no
    # elif val == 2:
    #     storm_index[j] = storm_no if time_since_rain[j+1] != 3 else None
    #     total[j] = storm_no
    # else:
    #     storm_index[j] = None
    #     total[j] = storm_no-1 if time_since_rain[j] >= 3 else storm_no