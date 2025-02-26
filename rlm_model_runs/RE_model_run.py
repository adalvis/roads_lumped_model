"""
Author: Amanda Alvis
Date: 06/16/2020
Update: 05/03/2024
Purpose: Lumped model of road prism forced using hourly weather station data  
         at North Fork Toutle; Lat: 46.37194, Lon: -122.57778
         Data pulled from Mesonet (https://developers.synopticdata.com/mesonet/)
"""
#%%
#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

#Read in .csv of pre-grouped storms
data_df = pd.read_csv('/home/adalvis/github/roads_lumped_model/rlm_data/groupedStorms_ElkRock_10yr.csv', index_col='date')
data_df.index = pd.to_datetime(data_df.index)

#Length of storm in # of hourly time steps
timeStep_Hr = data_df.groupby('totalNo')['totalNo'].count().to_numpy()
#Depth of storm in mm
groupedDepth = data_df.groupedDepth

deltaT = timeStep_Hr #time step
totalT = np.cumsum(deltaT) #total time
truck_pass_5 = [] #initialize empty list for truck passes
truck_pass_10 = []
truck_pass_20 = []

np.random.seed(2) #Use seed to ensure consistent results with each run

for i, time in enumerate(timeStep_Hr):
    truck_5 = 0
    truck_10 = 0
    truck_20 = 0
    if time/24 >= 1:
        day = int(time/24)
        frac_day = time/24 - int(time/24)
        for num in range(day):
            truck_20 += np.random.poisson(20,1).item() #double trucks
            truck_10 += np.random.poisson(10,1).item() #normal trucks
            truck_5 += np.random.poisson(5,1).item() #half trucks
        truck_20 += round(np.random.poisson(20,1).item()*frac_day) #double trucks
        truck_10 += round(np.random.poisson(10,1).item()*frac_day) #normal trucks
        truck_5 += round(np.random.poisson(5,1).item()*frac_day) #half trucks
    else:
        frac_day = time/24 - int(time/24)
        truck_20 = round(np.random.poisson(20,1).item()*frac_day) #double trucks
        truck_10 = round(np.random.poisson(10,1).item()*frac_day) #normal trucks
        truck_5 = round(np.random.poisson(5,1).item()*frac_day) #half trucks
    truck_pass_5.append(truck_5) #number of truck passes    
    truck_pass_10.append(truck_10) #number of truck passes
    truck_pass_20.append(truck_20) #number of truck passes


#Create new dataframe that's grouped into storm/interstorm time periods
storms_df = pd.DataFrame()
storms_df['time'] = totalT
storms_df['deltaT'] = deltaT
storms_df['day'] = np.divide(totalT,24).astype('int64')
storms_df['truck_pass_5'] = truck_pass_5
storms_df['truck_pass_10'] = truck_pass_10
storms_df['truck_pass_20'] = truck_pass_20
storms_df['stormNo'] = data_df.groupby('stormNo')['stormNo'].mean()
storms_df['intensity'] = data_df.groupby('stormNo')['intensity_mmhr'].mean() 
storms_df['q_mean'] = storms_df.intensity*2.77778e-7*4.57

day0 = data_df.index[0]
storms_df.set_index(pd.DatetimeIndex([day0+datetime.timedelta(hours=time) 
    for time in storms_df.time]), inplace=True)
#%%===========================DEFINE PHYSICAL CONSTANTS===========================
# L = representative segment of road, m
# S = m/m
# tau_c = N/m^2
L, rho_w, rho_s, g, S, tau_c, d50, d95 = [4.57, 1000, 2650, 
                                          9.81, 0.03, 0.052,
                                          1.8e-5, 0.0275]
#%%===========================DEFINE LAYER CONSTANTS===========================
# h_s = depth of surfacing
# f_sf, f_sc = fractions of fine/coarse material in ballast
# h_b = depth of surfacing
# f_bf, f_bc = fractions of fine/coarse material in ballast
h_s, f_sf, f_sc = [0.23, 0.275, 0.725]
h_b, f_bf, f_br = [2, 0.20, 0.80]

#%%===========================DEFINE PUMPING/CRUSHING RATES===========================
# k_cs = crushing constant for surfacing, m/truck pass
# k_cb = crushing constant for ballast, m/truck pass
# u_ps = pumping constant for surfacing, m/truck pass
#   (2.14e-5m^3/4.57 m^2)  
#   6 tires * 0.225 m width * 0.005 m length * 3.175e-3 m treads
# u_pb = pumping constant for ballast, m/truck pass
# e = fraction of coarse material, -
k_cs, k_cb, u_ps, u_pb, e = [1e-7, 1e-7, 5e-7, 1e-7, 0.725]

#%%===========================GROUP RAINFALL DATA===========================
#Group data_df.intensity_mmhr into intensity "buckets" and count the values in each "bucket"
data_ser = data_df.groupby(['stormNo','intensity_mmhr'])['intensity_mmhr'].count()

#Rename the grouped data "num" to signify # of values in "bucket"
data_ser.rename('num', inplace=True)

#Create a new df from data_ser using reset_index
int_tip_df = data_ser.reset_index(level=1)
#Reset index again s.t. we have only #s for index
int_tip_df.reset_index(inplace=True)

int_tip_df['tot'] = int_tip_df.groupby('stormNo')['num'].transform('sum')
int_tip_df['frac'] = int_tip_df.num/int_tip_df.tot
int_tip_df['storm_dur'] = int_tip_df['tot']

#Drop na values
int_tip_df.drop([0], inplace=True)


#%%===========================INITIALIZE ARRAYS===========================
#=================len(storms_df)====================
dS_f_5 = np.zeros(len(storms_df))
S_f_init_5 = np.zeros(len(storms_df))
S_f_5 = np.zeros(len(storms_df))
S_s_5 = np.zeros(len(storms_df))
S_sc_5 = np.zeros(len(storms_df))
S_sf_5 = np.zeros(len(storms_df))
S_b_5 = np.zeros(len(storms_df))
S_bc_5 = np.zeros(len(storms_df))
S_bf_5 = np.zeros(len(storms_df))
Hs_out_5 = np.zeros(len(storms_df))
q_ps_5 = np.zeros(len(storms_df))
q_pb_5 = np.zeros(len(storms_df))
q_cs_5 = np.zeros(len(storms_df))
q_cb_5 = np.zeros(len(storms_df))
sed_added_5 = np.zeros(len(storms_df))
q_s_avg_5 = np.zeros(len(storms_df))
n_tp_5 = storms_df.truck_pass_5.to_numpy()
sed_cap_5 = np.zeros(len(storms_df))
ref_trans_5 = np.zeros(len(storms_df))
q_ref_avg_5 = np.zeros(len(storms_df))

dS_f_10 = np.zeros(len(storms_df))
S_f_init_10 = np.zeros(len(storms_df))
S_f_10 = np.zeros(len(storms_df))
S_s_10 = np.zeros(len(storms_df))
S_sc_10 = np.zeros(len(storms_df))
S_sf_10 = np.zeros(len(storms_df))
S_b_10 = np.zeros(len(storms_df))
S_bc_10 = np.zeros(len(storms_df))
S_bf_10 = np.zeros(len(storms_df))
Hs_out_10 = np.zeros(len(storms_df))
q_ps_10 = np.zeros(len(storms_df))
q_pb_10 = np.zeros(len(storms_df))
q_cs_10 = np.zeros(len(storms_df))
q_cb_10 = np.zeros(len(storms_df))
sed_added_10 = np.zeros(len(storms_df))
q_s_avg_10 = np.zeros(len(storms_df))
n_tp_10 = storms_df.truck_pass_10.to_numpy()
sed_cap_10 = np.zeros(len(storms_df))
ref_trans_10 = np.zeros(len(storms_df))
q_ref_avg_10 = np.zeros(len(storms_df))

dS_f_20 = np.zeros(len(storms_df))
S_f_init_20 = np.zeros(len(storms_df))
S_f_20 = np.zeros(len(storms_df))
S_s_20 = np.zeros(len(storms_df))
S_sc_20 = np.zeros(len(storms_df))
S_sf_20 = np.zeros(len(storms_df))
S_b_20 = np.zeros(len(storms_df))
S_bc_20 = np.zeros(len(storms_df))
S_bf_20 = np.zeros(len(storms_df))
Hs_out_20 = np.zeros(len(storms_df))
q_ps_20 = np.zeros(len(storms_df))
q_pb_20 = np.zeros(len(storms_df))
q_cs_20 = np.zeros(len(storms_df))
q_cb_20 = np.zeros(len(storms_df))
sed_added_20 = np.zeros(len(storms_df))
q_s_avg_20 = np.zeros(len(storms_df))
n_tp_20 = storms_df.truck_pass_20.to_numpy()
sed_cap_20 = np.zeros(len(storms_df))
ref_trans_20 = np.zeros(len(storms_df))
q_ref_avg_20 = np.zeros(len(storms_df))

r_storm = np.zeros(len(storms_df))
q_storm = np.zeros(len(storms_df))
t = storms_df.deltaT.to_numpy()

#=================len(int_tip_df)====================
f_s_5 = np.zeros(len(int_tip_df))
n_f_5 = np.zeros(len(int_tip_df))
n_t_5 = np.zeros(len(int_tip_df))
q_s_5 = np.zeros(len(int_tip_df))
tau_e_5 = np.zeros(len(int_tip_df))
q_ref_5 = np.zeros(len(int_tip_df))
water_depth_5 = np.zeros(len(int_tip_df))
tau_5 = np.zeros(len(int_tip_df))

f_s_10 = np.zeros(len(int_tip_df))
n_f_10 = np.zeros(len(int_tip_df))
n_t_10 = np.zeros(len(int_tip_df))
q_s_10 = np.zeros(len(int_tip_df))
tau_e_10 = np.zeros(len(int_tip_df))
q_ref_10 = np.zeros(len(int_tip_df))
water_depth_10 = np.zeros(len(int_tip_df))
tau_10 = np.zeros(len(int_tip_df))

f_s_20 = np.zeros(len(int_tip_df))
n_f_20 = np.zeros(len(int_tip_df))
n_t_20 = np.zeros(len(int_tip_df))
q_s_20 = np.zeros(len(int_tip_df))
tau_e_20 = np.zeros(len(int_tip_df))
q_ref_20 = np.zeros(len(int_tip_df))
water_depth_20 = np.zeros(len(int_tip_df))
tau_20 = np.zeros(len(int_tip_df))

r_avg = np.zeros(len(int_tip_df))
q = np.zeros(len(int_tip_df))
rainfall = int_tip_df.intensity_mmhr.to_numpy()
frac = int_tip_df.frac.to_numpy()
stormNo = int_tip_df.stormNo.to_numpy()
t_storm = int_tip_df.groupby('stormNo')['storm_dur'].mean().to_numpy()


#%%===========================INITIALIZE DEPTHS n = 5===========================
S_f_init_5[0] = 0.0275
S_f_5[0] = 0.0275
S_s_5[0] = h_s*(f_sf + f_sc)
S_sc_5[0] = h_s*(f_sc)
S_sf_5[0] = h_s*(f_sf)
S_b_5[0] = h_b*(f_bf + f_br)
S_bc_5[0] = h_b*(f_br)
S_bf_5[0] = h_b*(f_bf)
n_t_5[0] = 0.4
n_c_5 = 0.4

for j, storm in enumerate(storms_df.stormNo):
    if j == 0:
        continue
    else:
        q_ps_5[j] = u_ps*(S_sf_5[j-1]/S_s_5[j-1])*n_tp_5[j]/(t[j]*3600)
        q_pb_5[j] = u_pb*(S_bf_5[j-1]/S_b_5[j-1])*n_tp_5[j]/(t[j]*3600)
        q_cs_5[j] = k_cs*(S_sc_5[j-1]/S_s_5[j-1])*n_tp_5[j]/(t[j]*3600)
        q_cb_5[j] = k_cb*(S_bc_5[j-1]/S_b_5[j-1])*n_tp_5[j]/(t[j]*3600)
        S_bc_5[j] = S_bc_5[j-1] - q_cb_5[j]*(t[j]*3600)
        S_sc_5[j] = S_sc_5[j-1] - q_cs_5[j]*(t[j]*3600)
        S_bf_5[j] = S_bf_5[j-1] + q_cb_5[j]*(t[j]*3600) - q_pb_5[j]*(t[j]*3600)
        S_sf_5[j] = S_sf_5[j-1] + q_cs_5[j]*(t[j]*3600) - q_ps_5[j]*(t[j]*3600) + q_pb_5[j]*(t[j]*3600)
        S_s_5[j] = S_sc_5[j] + S_sf_5[j]
        S_b_5[j] = S_bc_5[j] + S_bf_5[j]

    if d95 >= S_f_5[j-1]:
        sed_added_5[j] = (q_ps_5[j]*(t[j]*3600.))/(1-e)
        S_f_init_5[j] = S_f_5[j-1] + sed_added_5[j]
    else:
        sed_added_5[j] = q_ps_5[j]*(t[j]*3600.)
        S_f_init_5[j] = S_f_5[j-1] + sed_added_5[j]
#===========================BEGIN INTEGRATE OVER qs n = 5===========================
    for k, val in enumerate(stormNo):
        if val == storm:
            if k == 0:
                continue
            else:
                q[k] = rainfall[k]*2.77778e-7*L 
                            
                if q[k] > 0:
                    #Determine Manning's GRAIN roughness
                    n_f_5[k] = 0.0026*q[k]**(-0.274)#*(S_f_init[j]/d95) #Based on Emmett (1970) Series 8 Lab Data
                    
                    if S_f_init_5[j] <= d95:
                        #Determine TOTAL Manning's roughness & partitioning ratio
                        n_t_5[k] = n_c_5 + (S_f_init_5[j]/d95)*(n_f_5[k]-n_c_5)
                        f_s_5[k] = (n_f_5[k]/n_t_5[k])**(1.5)*(S_f_init_5[j]/d95)
                    else: 
                        n_t_5[k] = n_f_5[k]              
                        f_s_5[k] = (n_f_5[k]/n_t_5[k])**(1.5)                 
                else:
                    n_f_5[k] = n_f_5[k-1]
                    n_t_5[k] = n_t_5[k-1]

                #Calculate water depth assuming uniform overland flow
                water_depth_5[k] = ((n_t_5[k]*q[k])/(S**(1/2)))**(3/5)

                tau_5[k] = rho_w*g*water_depth_5[k]*S
                tau_e_5[k] = tau_5[k]*f_s_5[k]

                #Calculate sediment transport rate
                if (tau_e_5[k]-tau_c) >= 0:
                    q_s_5[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*(tau_e_5[k]-tau_c)**(2.457))
                else:
                    q_s_5[k] = 0

                #Calculate reference transport 
                if (tau_5[k]-tau_c) >=0:
                    q_ref_5[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*(tau_5[k]-tau_c)**(2.457))
                else:
                    q_ref_5[k] = 0
            q_storm[j] += q[k]*frac[k]
            q_s_avg_5[j] += q_s_5[k]*frac[k]
            q_ref_avg_5[j] += q_ref_5[k]*frac[k]

#===========================END INTEGRATE OVER qs===========================
            
    sed_cap_5[j] = q_s_avg_5[j]*t_storm[j]*3600/L
    ref_trans_5[j] = q_ref_avg_5[j]*t_storm[j]*3600/L

    Hs_out_5[j] = np.minimum(sed_added_5[j]+S_f_5[j-1], sed_cap_5[j])
    dS_f_5[j] = sed_added_5[j] - Hs_out_5[j]
    S_f_5[j] = S_f_5[j-1] + dS_f_5[j]

#%%===========================INITIALIZE DEPTHS n = 10===========================
S_f_init_10[0] = 0.0275
S_f_10[0] = 0.0275
S_s_10[0] = h_s*(f_sf + f_sc)
S_sc_10[0] = h_s*(f_sc)
S_sf_10[0] = h_s*(f_sf)
S_b_10[0] = h_b*(f_bf + f_br)
S_bc_10[0] = h_b*(f_br)
S_bf_10[0] = h_b*(f_bf)
n_t_10[0] = 0.4
n_c_10 = 0.4

for j, storm in enumerate(storms_df.stormNo):
    if j == 0:
        continue
    else:
        q_ps_10[j] = u_ps*(S_sf_10[j-1]/S_s_10[j-1])*n_tp_10[j]/(t[j]*3600)
        q_pb_10[j] = u_pb*(S_bf_10[j-1]/S_b_10[j-1])*n_tp_10[j]/(t[j]*3600)
        q_cs_10[j] = k_cs*(S_sc_10[j-1]/S_s_10[j-1])*n_tp_10[j]/(t[j]*3600)
        q_cb_10[j] = k_cb*(S_bc_10[j-1]/S_b_10[j-1])*n_tp_10[j]/(t[j]*3600)
        S_bc_10[j] = S_bc_10[j-1] - q_cb_10[j]*(t[j]*3600)
        S_sc_10[j] = S_sc_10[j-1] - q_cs_10[j]*(t[j]*3600)
        S_bf_10[j] = S_bf_10[j-1] + q_cb_10[j]*(t[j]*3600) - q_pb_10[j]*(t[j]*3600)
        S_sf_10[j] = S_sf_10[j-1] + q_cs_10[j]*(t[j]*3600) - q_ps_10[j]*(t[j]*3600) + q_pb_10[j]*(t[j]*3600)
        S_s_10[j] = S_sc_10[j] + S_sf_10[j]
        S_b_10[j] = S_bc_10[j] + S_bf_10[j]

    if d95 >= S_f_10[j-1]:
        sed_added_10[j] = (q_ps_10[j]*(t[j]*3600.))/(1-e)
        S_f_init_10[j] = S_f_10[j-1] + sed_added_10[j]
    else:
        sed_added_10[j] = q_ps_10[j]*(t[j]*3600.)
        S_f_init_10[j] = S_f_10[j-1] + sed_added_10[j]
#===========================BEGIN INTEGRATE OVER qs n = 10===========================
    for k, val in enumerate(stormNo):
        if val == storm:
            if k == 0:
                continue
            else:
                q[k] = rainfall[k]*2.77778e-7*L 
                            
                if q[k] > 0:
                    #Determine Manning's GRAIN roughness
                    n_f_10[k] = 0.0026*q[k]**(-0.274)#*(S_f_init[j]/d95) #Based on Emmett (1970) Series 8 Lab Data
                    
                    if S_f_init_10[j] <= d95:
                        #Determine TOTAL Manning's roughness & partitioning ratio
                        n_t_10[k] = n_c_10 + (S_f_init_10[j]/d95)*(n_f_10[k]-n_c_10)
                        f_s_10[k] = (n_f_10[k]/n_t_10[k])**(1.5)*(S_f_init_10[j]/d95)
                    else: 
                        n_t_10[k] = n_f_10[k]              
                        f_s_10[k] = (n_f_10[k]/n_t_10[k])**(1.5)                 
                else:
                    n_f_10[k] = n_f_10[k-1]
                    n_t_10[k] = n_t_10[k-1]

                #Calculate water depth assuming uniform overland flow
                water_depth_10[k] = ((n_t_10[k]*q[k])/(S**(1/2)))**(3/5)

                tau_10[k] = rho_w*g*water_depth_10[k]*S
                tau_e_10[k] = tau_10[k]*f_s_10[k]

                #Calculate sediment transport rate
                if (tau_e_10[k]-tau_c) >= 0:
                    q_s_10[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*(tau_e_10[k]-tau_c)**(2.457))
                else:
                    q_s_10[k] = 0

                #Calculate reference transport 
                if (tau_10[k]-tau_c) >=0:
                    q_ref_10[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*(tau_10[k]-tau_c)**(2.457))
                else:
                    q_ref_10[k] = 0
            q_storm[j] += q[k]*frac[k]
            q_s_avg_10[j] += q_s_10[k]*frac[k]
            q_ref_avg_10[j] += q_ref_10[k]*frac[k]

#===========================END INTEGRATE OVER qs===========================
        sed_cap_10[j] = q_s_avg_10[j]*t_storm[j]*3600/L
        ref_trans_10[j] = q_ref_avg_10[j]*t_storm[j]*3600/L

        Hs_out_10[j] = np.minimum(sed_added_10[j]+S_f_10[j-1], sed_cap_10[j])
        dS_f_10[j] = sed_added_10[j] - Hs_out_10[j]
        S_f_10[j] = S_f_10[j-1] + dS_f_10[j]



#%%===========================INITIALIZE DEPTHS n = 20===========================
S_f_init_20[0] = 0.0275
S_f_20[0] = 0.0275
S_s_20[0] = h_s*(f_sf + f_sc)
S_sc_20[0] = h_s*(f_sc)
S_sf_20[0] = h_s*(f_sf)
S_b_20[0] = h_b*(f_bf + f_br)
S_bc_20[0] = h_b*(f_br)
S_bf_20[0] = h_b*(f_bf)
n_t_20[0] = 0.4
n_c_20 = 0.4

for j, storm in enumerate(storms_df.stormNo):
    if j == 0:
        continue
    else:
        q_ps_20[j] = u_ps*(S_sf_20[j-1]/S_s_20[j-1])*n_tp_20[j]/(t[j]*3600)
        q_pb_20[j] = u_pb*(S_bf_20[j-1]/S_b_20[j-1])*n_tp_20[j]/(t[j]*3600)
        q_cs_20[j] = k_cs*(S_sc_20[j-1]/S_s_20[j-1])*n_tp_20[j]/(t[j]*3600)
        q_cb_20[j] = k_cb*(S_bc_20[j-1]/S_b_20[j-1])*n_tp_20[j]/(t[j]*3600)
        S_bc_20[j] = S_bc_20[j-1] - q_cb_20[j]*(t[j]*3600)
        S_sc_20[j] = S_sc_20[j-1] - q_cs_20[j]*(t[j]*3600)
        S_bf_20[j] = S_bf_20[j-1] + q_cb_20[j]*(t[j]*3600) - q_pb_20[j]*(t[j]*3600)
        S_sf_20[j] = S_sf_20[j-1] + q_cs_20[j]*(t[j]*3600) - q_ps_20[j]*(t[j]*3600) + q_pb_20[j]*(t[j]*3600)
        S_s_20[j] = S_sc_20[j] + S_sf_20[j]
        S_b_20[j] = S_bc_20[j] + S_bf_20[j]

    if d95 >= S_f_20[j-1]:
        sed_added_20[j] = (q_ps_20[j]*(t[j]*3600.))/(1-e)
        S_f_init_20[j] = S_f_20[j-1] + sed_added_20[j]
    else:
        sed_added_20[j] = q_ps_20[j]*(t[j]*3600.)
        S_f_init_20[j] = S_f_20[j-1] + sed_added_20[j]

#===========================BEGIN INTEGRATE OVER qs n = 20===========================
    for k, val in enumerate(stormNo):
        if val == storm:
            if k == 0:
                continue
            else:
                q[k] = rainfall[k]*2.77778e-7*L 
                            
                if q[k] > 0:
                    #Determine Manning's GRAIN roughness
                    n_f_20[k] = 0.0026*q[k]**(-0.274)#*(S_f_init[j]/d95) #Based on Emmett (1970) Series 8 Lab Data
                    
                    if S_f_init_20[j] <= d95:
                        #Determine TOTAL Manning's roughness & partitioning ratio
                        n_t_20[k] = n_c_20 + (S_f_init_20[j]/d95)*(n_f_20[k]-n_c_20)
                        f_s_20[k] = (n_f_20[k]/n_t_20[k])**(1.5)*(S_f_init_20[j]/d95)
                    else: 
                        n_t_20[k] = n_f_20[k]              
                        f_s_20[k] = (n_f_20[k]/n_t_20[k])**(1.5)                 
                else:
                    n_f_20[k] = n_f_20[k-1]
                    n_t_20[k] = n_t_20[k-1]

                #Calculate water depth assuming uniform overland flow
                water_depth_20[k] = ((n_t_20[k]*q[k])/(S**(1/2)))**(3/5)

                tau_20[k] = rho_w*g*water_depth_20[k]*S
                tau_e_20[k] = tau_20[k]*f_s_20[k]

                #Calculate sediment transport rate
                if (tau_e_20[k]-tau_c) >= 0:
                    q_s_20[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*(tau_e_20[k]-tau_c)**(2.457))
                else:
                    q_s_20[k] = 0

                #Calculate reference transport 
                if (tau_20[k]-tau_c) >=0:
                    q_ref_20[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*(tau_20[k]-tau_c)**(2.457))
                else:
                    q_ref_20[k] = 0
            q_storm[j] += q[k]*frac[k]
            q_s_avg_20[j] += q_s_20[k]*frac[k]
            q_ref_avg_20[j] += q_ref_20[k]*frac[k]


    #===========================END INTEGRATE OVER qs===========================
            
        sed_cap_20[j] = q_s_avg_20[j]*t_storm[j]*3600/L
        ref_trans_20[j] = q_ref_avg_20[j]*t_storm[j]*3600/L

        Hs_out_20[j] = np.minimum(sed_added_20[j]+S_f_20[j-1], sed_cap_20[j])
        dS_f_20[j] = sed_added_20[j] - Hs_out_20[j]
        S_f_20[j] = S_f_20[j-1] + dS_f_20[j]


#%% Step 2!
#Add all numpy arrays to the Pandas dataframe
storms_df['qf1_5'] = q_ps_5
storms_df['qf2_5'] = q_pb_5
storms_df['S_f_5'] = S_f_5*1000
storms_df['S_s_5'] = S_s_5
storms_df['S_sc_5'] = S_sc_5
storms_df['S_sf_5'] = S_sf_5
storms_df['S_b_5'] = S_b_5
storms_df['S_bc_5'] = S_bc_5
storms_df['S_bf_5'] = S_bf_5
storms_df['Hs_out_5'] = Hs_out_5*1000
storms_df['sed_added_5'] = sed_added_5
storms_df['sed_cap_5'] = sed_cap_5*1000
storms_df['ref_trans_5'] = ref_trans_5*1000

storms_df['qf1_10'] = q_ps_10
storms_df['qf2_10'] = q_pb_10
storms_df['S_f_10'] = S_f_10*1000
storms_df['S_s_10'] = S_s_10
storms_df['S_sc_10'] = S_sc_10
storms_df['S_sf_10'] = S_sf_10
storms_df['S_b_10'] = S_b_10
storms_df['S_bc_10'] = S_bc_10
storms_df['S_bf_10'] = S_bf_10
storms_df['Hs_out_10'] = Hs_out_10*1000
storms_df['sed_added_10'] = sed_added_10
storms_df['sed_cap_10'] = sed_cap_10*1000
storms_df['ref_trans_10'] = ref_trans_10*1000

storms_df['qf1_20'] = q_ps_20
storms_df['qf2_20'] = q_pb_20
storms_df['S_f_20'] = S_f_20*1000
storms_df['S_s_20'] = S_s_20
storms_df['S_sc_20'] = S_sc_20
storms_df['S_sf_20'] = S_sf_20
storms_df['S_b_20'] = S_b_20
storms_df['S_bc_20'] = S_bc_20
storms_df['S_bf_20'] = S_bf_20
storms_df['Hs_out_20'] = Hs_out_20*1000
storms_df['sed_added_20'] = sed_added_20
storms_df['sed_cap_20'] = sed_cap_20*1000
storms_df['ref_trans_20'] = ref_trans_20*1000

storms_df['q_storm'] = q_storm
int_tip_df['q'] = q
# int_tip_df.to_csv('/home/adalvis/github/roads_lumped_model/rlm_data/int_tip_df_new.csv')
#%%
plt.close('all')
#Plot sediment transport capacity and actual transport over time
fig1, ax1 = plt.subplots(1, 3, figsize=(12,4), layout='tight')
storms_df.ref_trans_5.cumsum().plot(color = '#95190C', label='Cumulative reference\ntransport capacity', ax=ax1[0])
storms_df.ref_trans_10.cumsum().plot(color = '#95190C', label='Cumulative reference\ntransport capacity', ax=ax1[1])
storms_df.ref_trans_20.cumsum().plot(color = '#95190C', label='Cumulative reference\ntransport capacity', ax=ax1[2])
storms_df.Hs_out_5.cumsum().plot(linestyle='-', color='#D58936', label='Cumulative transport', ax=ax1[0], secondary_y=True)
storms_df.Hs_out_10.cumsum().plot(linestyle='-', color='#D58936', label='Cumulative transport', ax=ax1[1], secondary_y=True)
storms_df.Hs_out_20.cumsum().plot(linestyle='-', color='#D58936', label='Cumulative transport', ax=ax1[2], secondary_y=True)

for ax in ax1.flatten():
    ax.legend(loc="upper left")
    ax.right_ax.legend(loc="lower right")
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    ax.set(xlabel='Date', ylim=(0,675))
    ax.tick_params(axis='x', labelrotation=25)
    ax.right_ax.set_ylim(0,40)
    for tl in ax.get_yticklabels():
        tl.set_color('#95190C') 
    for tl in ax.right_ax.get_yticklabels():
        tl.set_color('#D58936') 

ax1[0].set_title(r'Mean $n_{\Delta t}$ = 5 tpd')
ax1[1].set_title(r'Mean $n_{\Delta t}$ = 10 tpd')
ax1[2].set_title(r'Mean $n_{\Delta t}$ = 20 tpd')
ax1[0].set_ylabel('Cumulative sediment depth (mm)', labelpad=5, color='#95190C')
# ax1[2].right_ax.set_ylabel('Cumulative sediment depth (mm)', rotation=270, labelpad=15, color='#D58936')

plt.tight_layout()
# plt.savefig(r'/home/adalvis/github/roads_lumped_model/savefigs/Fig4_triPanel.png', dpi=600)
plt.show()

#%%
#Plot fine sediment storage over time
fig2, ax2 = plt.subplots(figsize=(6,4))
storms_df.plot(y='S_f_20', ax=ax2, color = '#610345', label=r'Mean $n_{\Delta t}$ = 20 tpd')
storms_df.plot(y='S_f_10', ax=ax2, color = '#107E7D', label=r'Mean $n_{\Delta t}$ = 10 tpd')
storms_df.plot(y='S_f_5', ax=ax2, color = '#D5573B', label=r'Mean $n_{\Delta t}$ = 5 tpd')
plt.xlabel('Date')
plt.ylabel('Fine sediment storage, $S_a$ (mm)')
# plt.title('Fine sediment storage')
ax2.legend(loc="upper right")
ax2.set_ylim(23,29)
for label in ax2.get_xticklabels():
    label.set_horizontalalignment('center')
plt.xticks(rotation=0)
plt.tight_layout()
# plt.savefig(r'/home/adalvis/github/roads_lumped_model/savefigs/Fig3_three.png', dpi=600)
plt.show()

#%%
#Plot fine sediment storage and actual transport over time
# fig3, ax3 = plt.subplots(figsize=(7,3))
# storms_df.plot(y='S_f', ax=ax3, color = 'mediumseagreen', legend=False, 
#     label='Fine storage')
# storms_df.plot(y='Hs_out', ax=ax3, color = '#442766', legend=False, 
#     label='Actual transport', alpha=0.75)
# ax3.set_xlabel('Date')
# ax3.set_ylabel(r'Sediment depth $(mm)$')
# fig3.legend(loc="upper right", bbox_to_anchor=(1,1), 
#     bbox_transform=ax3.transAxes)
# plt.tight_layout()
# plt.show()
# #%%
#Plot surfacing storage over time
fig4, ax4 = plt.subplots(figsize=(6,4))
storms_df.plot(y='S_s_20', ax=ax4, color = '#610345', label=r'Mean $n_{\Delta t}$ = 20 tpd')
storms_df.plot(y='S_s_10', ax=ax4, color = '#107E7D', label=r'Mean $n_{\Delta t}$ = 10 tpd')
storms_df.plot(y='S_s_5', ax=ax4, color = '#D5573B', label=r'Mean $n_{\Delta t}$ = 5 tpd')
storms_df.plot(y='S_sc_5', ax=ax4, color = '#7B583D', legend=False, 
    label='Coarse surfacing n=5')
storms_df.plot(y='S_sc_10', ax=ax4, color = '#7B583D', legend=False, 
    label='Coarse surfacing n=10')
storms_df.plot(y='S_sc_20', ax=ax4, color = '#7B583D', legend=False, 
    label='Coarse surfacing, n=20')
storms_df.plot(y='S_sf_5', ax=ax4, color = '#587A81', legend=False, 
    label='Fine surfacing, n=5')
storms_df.plot(y='S_sf_10', ax=ax4, color = '#587A81', legend=False, 
    label='Fine surfacing, n=10')
storms_df.plot(y='S_sf_20', ax=ax4, color = '#587A81', legend=False, 
    label='Fine surfacing,  n=20')
ax4.set_ylabel(r'Surfacing storage, $S_s$ $(m)$')
ax4.set_ylim(0.05,0.2305)
ax4.tick_params(axis='x', labelrotation=0)
for label in ax4.get_xticklabels():
    label.set_horizontalalignment('center')
plt.legend()
# ax4.set_title('Surfacing storage')
# fig4.legend(loc="upper right", bbox_to_anchor=(1,1), 
#     bbox_transform=ax4.transAxes)
plt.xlabel('Date')
plt.tight_layout()
# plt.savefig(r'C:/Users/Amanda/Documents/GitHub/roads_lumped_model/rlm_output/hourly/surfacing/Surf_%s.png' %S_f_init[0], dpi=300)
plt.show()
#%%
#Plot ballast storage over time
fig5, ax5 = plt.subplots(figsize=(6,4))
storms_df.plot(y='S_b_20', ax=ax5, color = '#610345', label=r'Mean $n_{\Delta t}$ = 20 tpd')
storms_df.plot(y='S_b_10', ax=ax5, color = '#107E7D', label=r'Mean $n_{\Delta t}$ = 10 tpd')
storms_df.plot(y='S_b_5', ax=ax5, color = '#D5573B', label=r'Mean $n_{\Delta t}$ = 5 tpd')

ax5.set_ylim(1.99875, 2.00005)
ax5.tick_params(axis='x', labelrotation=0)
for label in ax5.get_xticklabels():
    label.set_horizontalalignment('center')
# storms_df.plot(y='S_bc', ax=ax5, color = '#AB6B51', legend=False, 
#     label='Coarse ballast')
# storms_df.plot(y='S_bf', ax=ax5, color = '#39918C', legend=False, 
#     label='Fine ballast')
plt.legend()
plt.xlabel('Date')
plt.ylabel(r'Ballast storage, $S_b$ $(m)$')
# fig5.legend(loc="upper right", bbox_to_anchor=(1,1), 
#     bbox_transform=ax5.transAxes)
# plt.title('Ballast storage')
plt.tight_layout()
# plt.savefig(r'C:/Users/Amanda/Documents/GitHub/roads_lumped_model/rlm_output/hourly/ballast/Bal_%s.png' %S_f_init[0], dpi=300)
plt.show()
#%%
#Subset data by water year
years=storms_df.groupby(storms_df.index.year).count().index.to_numpy()
yr_5 = np.zeros(len(years)-1)
yr_10 = np.zeros(len(years)-1)
yr_20 = np.zeros(len(years)-1)
for i, year in enumerate(years):
    if i < len(years)-1:
        yr_5[i] = (storms_df.Hs_out_5[str(year)+'-07-01':str(year+1)+'-06-30'].sum())/1000
        yr_10[i] = (storms_df.Hs_out_10[str(year)+'-07-01':str(year+1)+'-06-30'].sum())/1000
        yr_20[i] = (storms_df.Hs_out_20[str(year)+'-07-01':str(year+1)+'-06-30'].sum())/1000

#Multiply Hs_out
sed_area_5 = np.multiply(yr_5, L)
sed_load_5 = np.multiply(sed_area_5, rho_s)

sed_area_10 = np.multiply(yr_10, L)
sed_load_10 = np.multiply(sed_area_10, rho_s)

sed_area_20 = np.multiply(yr_20, L)
sed_load_20 = np.multiply(sed_area_20, rho_s)

width = 0.25

ticks = years[1:len(years)]
fig6, ax6 = plt.subplots(figsize=(6,4))
plt.bar(years[1:len(years)]-width, sed_load_20, width, color = '#610345', edgecolor='k', label=r'Mean $n_{\Delta t}$ = 20 tpd')
plt.bar(years[1:len(years)], sed_load_10, width, color = '#107E7D', edgecolor='k', label=r'Mean $n_{\Delta t}$ = 10 tpd')
plt.bar(years[1:len(years)]+width, sed_load_5, width, color = '#D5573B', edgecolor='k', label=r'Mean $n_{\Delta t}$ = 5 tpd')
plt.xlabel('Water year')
plt.ylabel(r'Mass per meter of road (kg/m)')
# plt.title('Yearly sediment load per meter of road')
plt.legend()
plt.xticks(range(ticks[0],ticks[len(ticks)-1]+1), ticks, rotation=0)
plt.tight_layout()
# plt.savefig(r'/home/adalvis/github/roads_lumped_model/savefigs/Fig5_three.png', dpi=600)
plt.show()

# sed_sum_m = storms_df.sed_added.sum()-(storms_df.Hs_out.sum()/1000)
# sed_sum_kg_m = sed_sum_m*rho_s*L
# f = ((storms_df.S_f[len(storms_df)-1]-storms_df.S_f[0])/1000)*rho_s*L

# if round(f) == round(sed_sum_kg_m):
#     print('\nThe mass balance is fine.')
# else:
#     print('\nThe mass balance is off.')

# total_out_kg = (storms_df.Hs_out.sum()/1000)*rho_s*L
# print("\nTotal amount of sediment transported:", round(total_out_kg), "kg/m")

#Takes forever to run, hence down here.
# ticklabels = [item.strftime('%Y') for item in df_day.index[::366*2]]

# fig, ax = plt.subplots(figsize=(13,5))
# df_day.plot(y='truck_pass', ax=ax, color = '#8a0c80', legend=False, label='Truck passes', 
#             kind='bar', width=7)
# ax.set_xlabel('Date', fontsize=14, fontweight='bold')
# ax.set_ylabel('Truck passes', fontsize=14, fontweight='bold')
# ax.grid(False)

# ax1 = ax.twinx()
# df_day.plot(y='storm_depth', ax=ax1, color='#0c3c8a', legend=False, label='Storm depth', kind='bar', width=7)
# ax1.set_ylabel(r'Storm depth $(mm)$', fontsize=14, fontweight='bold')
# ax1.invert_yaxis()
# ax1.grid(False)

# fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
# ax.set_xticks(np.arange(0,366*2*len(ticklabels),366*2))
# ax.set_xticklabels(ticklabels, rotation=45)
# plt.tight_layout()
# #plt.savefig(r'C:\Users\Amanda\Desktop\Rainfall_Truck.png', dpi=300)
# plt.show()
# %%
