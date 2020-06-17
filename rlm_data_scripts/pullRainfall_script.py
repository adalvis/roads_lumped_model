import requests
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta

mesonetToken = '081d3a79e9304e25820be19e856bb049'

#Initialize lists to store data
date_start = str(201810010000)
date_end = str(201909300000)
stn_id = 'SRBW1' #North Fork Toutle; Lat: 46.37194, Lon: -122.57778
#stn_id = 'EKRW1' #Elk Rock; Lat: 46.312222, Lon: -122.385556


#Call API
r = requests.get('https://api.synopticdata.com/v2/stations/precip'+
                '?token='+mesonetToken+'&stid='+stn_id+'&start='+date_start+'&end='+
                date_end+'&obtimezone=UTC&pmode=intervals&interval=hour')

#Load API response as JSON
try:
    d = json.loads(r.text)
except:
    print('Error thrown\n')

#Access precip value dictionary in JSON file
dict = d['STATION'][0]['OBSERVATIONS']['precipitation']

#Get precip values
precip = [dict[i]['total'] for i in range(len(dict))]

#Get time data
time = [dict[i]['first_report'] for i in range(len(dict))]

#Create Pandas dataframe
rain_df = pd.DataFrame({'date':time, 'intensity_mmhr':precip})

#Convert date to normal datetime value in YYYY-mm-dd HH:mm:ss
rain_ser = pd.to_datetime(rain_df['date']).dt.tz_convert(None)

#Do some data manipulation to be able to interpolate dates
    #Convert dates to integers
rain = pd.Series(rain_ser.values.astype('int64'))
    #Make sure NaT values are NaN
rain[rain_ser.isnull()] = np.nan
    #Interpolate integers, then convert back to datetime
rain_df['date'] = pd.to_datetime(rain.interpolate())

#Set the index of the dataframe to be the date and drop the date column
rain_df = rain_df.set_index('date', drop=True)

rain_df.fillna(0.0, inplace=True)

#Save file as .csv
rain_df.to_csv(r'C:/Users/Amanda/Documents/GitHub/'+
               'roads_lumped_model/rlm_output/NFtoutle_rain.csv')