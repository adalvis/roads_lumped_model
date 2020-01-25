"""
Author: Amanda Manaster
Date: 01/09/2020
Purpose:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

#%%
storm_depth = []
rainfall_rate = []
delta_t = []
storm_length = []
truck_pass = []
total_t = []

model_end = 876000 #20 yrs = 175200 hrs; 100 yrs = 876000 hrs

#%%
np.random.seed(1) #Use seed to ensure consistent results with each run
time = 0 #model time; initial
T_b = 0  #average inter-storm duration; initial
T_r = 0  #average storm duration; initial
r = 0    #average intensity; initial

while time < model_end:
    truck = 0
    time_step = T_b+T_r
    
    if time_step/24 >= 1:
        day = int(time_step/24)
        frac_day = time_step/24 - int(time_step/24)
        
        for num in range(day):
            truck += np.random.randint(0,10)
            
        truck += round(np.random.randint(0,10)*frac_day)
    else:
        frac_day = time_step/24 - int(time_step/24)
        truck = round(np.random.randint(0,10)*frac_day)
    
    storm_length.append(T_r)             #length of storm
    storm_depth.append(r*T_r)            #depth of storm
    rainfall_rate.append(r)              #rate of rainfall
    delta_t.append(time_step)            #length of each time step; variable
    total_t.append(time)                 #model time
    truck_pass.append(truck)             #number of truck passes
    
    T_b = np.random.exponential(90.5)    #average inter-storm duration, hr
    T_r = np.random.exponential(2.705*2) #average storm duration, hr
    r = np.random.exponential(2)         #average intensity, mm/hr
    
    time += T_b+T_r
#%%
df = pd.DataFrame() #Create dataframe

df['time'] = total_t
df['delta_t'] = delta_t
df['day'] = np.divide(total_t,24).astype('int64')
df['storm_depth'] = storm_depth
df['rainfall_rate'] = rainfall_rate
df['storm_length'] = storm_length
df['truck_pass'] = truck_pass

day0 = datetime(2018, 10, 1)
df.set_index(pd.DatetimeIndex([day0+timedelta(hours=time) for time in df.time]), inplace=True)
#%%
df_day = df.resample('D').sum().fillna(0)
df_day.truck_pass = df_day.truck_pass.round()
df_day['day'] = np.arange(0, len(df_day), 1)
#%%
#ticklabels = [item.strftime('%Y') for item in df_day.index[::366*2]]
#
#fig, ax = plt.subplots(figsize=(13,5))
#df_day.plot(y='truck_pass', ax=ax, color = '#8a0c80', legend=False, label='Truck passes', 
#            kind='bar', width=7)
#ax.set_xlabel('Date', fontsize=14, fontweight='bold')
#ax.set_ylabel('Truck passes', fontsize=14, fontweight='bold')
#ax.grid(False)
#
#ax1 = ax.twinx()
#df_day.plot(y='storm_depth', ax=ax1, color='#0c3c8a', legend=False, label='Storm depth', kind='bar', width=7)
#ax1.set_ylabel(r'Storm depth $(mm)$', fontsize=14, fontweight='bold')
#ax1.invert_yaxis()
#ax1.grid(False)
#
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
#ax.set_xticks(np.arange(0,366*2*len(ticklabels),366*2))
#ax.set_xticklabels(ticklabels, rotation=45)
#plt.tight_layout()
##plt.savefig(r'C:\Users\Amanda\Desktop\Rainfall_Truck.png', dpi=300)
#
#plt.show()
#%%
#Define constants
L = 4.57 #representative segment of road, m
rho_w = 1000 #kg/m^3
rho_s = 2650 #kg/m^3
g = 9.81 #m/s^2
S = 0.0825 #m/m; 8% long slope, 2% lat slope
tau_c = 0.0 #N/m^2; assuming d50 is approx. 0.0580 mm; value from https://pubs.usgs.gov/sir/2008/5093/table7.html
d50 = 6.25e-5 #m
d95 = 0.055 #m
n_f = 0.0475*(d50)**(1/6) #approx Manning's n total
#%%
#define constants
h_s = 0.23
f_sf = 0.275
f_sc = 0.725

h_b = 2
f_bf = 0.20
f_br = 0.80

#The following four constants can be adjusted based on observations
kas = 1.37e-8 #crushing constant... value is easily changeable
kab = 1.0e-8
u_p = 4.69e-6 #m (2.14e-5m^3/4.57 m^2)  6 tires * 0.225 m width * 0.005 m length * 3.175e-3 m treads
u_f = 2.345e-6 #m
p = 0.20 #[-] (Applied Hydrogeology 3rd Ed. by C.W. Fetter, Table 3.4)
#%%
df_storage = pd.DataFrame()

df_storage['time'] = total_t
df_storage['day'] = np.divide(total_t,24).astype('int64')
day0 = datetime(2018, 10, 1)
df_storage.set_index(pd.DatetimeIndex([day0+timedelta(hours=time) for time in df_storage.time]), inplace=True)
#%%
#Step 1!
#Initialize numpy arrays for calculations
dS_f = np.zeros(len(df))
S_f = np.zeros(len(df))
S_s = np.zeros(len(df))
S_sc = np.zeros(len(df))
S_sf = np.zeros(len(df))
S_b = np.zeros(len(df))
S_bc = np.zeros(len(df))
S_bf = np.zeros(len(df))
Hs_out = np.zeros(len(df))
q_s = np.zeros(len(df))
h_f = np.zeros(len(df))
k_s = np.zeros(len(df))
H = np.zeros(len(df))
tau = np.zeros(len(df))
shear_stress = np.zeros(len(df))
f_s = np.zeros(len(df))
n_c = np.zeros(len(df))
n_t = np.zeros(len(df))

n_tp = df.truck_pass.to_numpy()
t = df.delta_t.to_numpy()
t_storm = df.storm_length.to_numpy()
rainfall = df.rainfall_rate.to_numpy()

q_f1 = np.zeros(len(df))
q_f2 = np.zeros(len(df))
q_as = np.zeros(len(df))
q_ab = np.zeros(len(df))
sed_added = np.zeros(len(df))
sed_cap = np.zeros(len(df))
value = np.zeros(len(df))

#Initial conditions for fines, surfacing, ballast
h_f[0] = (1-p)*(d95/2)
n_c[0] = 0.0475*(d95-h_f[0])**(1/6)
n_t[0] = n_f+n_c[0]
f_s[0] = (n_f/n_t[0])**(1.5)
S_f[0] = 0.0005
S_s[0] = h_s*(f_sf + f_sc)
S_sc[0] = h_s*(f_sc)
S_sf[0] = h_s*(f_sf)
S_b[0] = h_b*(f_bf + f_br)
S_bc[0] = h_b*(f_br)
S_bf[0] = h_b*(f_bf)
#%% 
#Step 2!
for i in range(1, len(df)):
    q_f1[i] = u_p*(S_sf[i-1]/S_s[i-1])*n_tp[i]/(t[i]*3600)
    q_f2[i] = u_f*(S_bf[i-1]/S_b[i-1])*n_tp[i]/(t[i]*3600)
    q_as[i] = kas*(S_sc[i-1]/S_s[i-1])*n_tp[i]/(t[i]*3600)
    q_ab[i] = kab*(S_bc[i-1]/S_b[i-1])*n_tp[i]/(t[i]*3600)
    
    S_bc[i] = S_bc[i-1] - q_ab[i]*(t[i]*3600)
    S_sc[i] = S_sc[i-1] - q_as[i]*(t[i]*3600)
    
    S_bf[i] = S_bf[i-1] + q_ab[i]*(t[i]*3600) - q_f2[i]*(t[i]*3600)
    S_sf[i] = S_sf[i-1] + q_as[i]*(t[i]*3600) - q_f1[i]*(t[i]*3600) + q_f2[i]*(t[i]*3600)
        
    S_s[i] = S_sc[i] + S_sf[i]
    S_b[i] = S_bc[i] + S_bf[i]
    
    h_f[i] = (1-p)*(q_f1[i]*(t[i]*3600)) + h_f[i-1] if d95 > h_f[i-1] else q_f1[i]*(t[i]*3600) + h_f[i-1] #porosity is only for added sediment!
    
    if d95 > h_f[i]:
        k_s[i] = d95 - h_f[i]
        n_c[i] = 0.0475*(k_s[i])**(1/6)
    else:
        n_c[i] = 0
    
    n_t[i] = n_f+n_c[i]
    
    f_s[i] = (n_f/n_t[i])**(1.5)
        
    #Calculate water depth assuming uniform overland flow
    H[i] = ((n_t[i]*(rainfall[i]/3.6e6)*L)/(S**(1/2)))**(3/5)
    
    tau[i] = rho_w*g*H[i]*S
    
    #Calculate shear stress
    shear_stress[i] = tau[i]*f_s[i]
    
    #Calculate sediment transport rate
    if (shear_stress[i]-tau_c) >= 0:
        q_s[i] = ((10**(-4.348))/(rho_s*((d50)**(0.811))))*(shear_stress[i]-tau_c)**(2.457)/L
    else:
        q_s[i] = 0

    #Create a condition column based on sediment transport capacity vs sediment supply
    sed_added[i] = (1-p)*q_f1[i]*(t[i]*3600.) if d95 > S_f[i-1] else q_f1[i]*(t[i]*3600.) #change to be sediment added and use this to calculate value
    sed_cap[i] = q_s[i]*(t_storm[i]*3600.)
    value[i] = (sed_added[i]-sed_cap[i])
        
    if value[i] < 0:
        Hs_out[i] = np.minimum(sed_added[i]+S_f[i-1], sed_cap[i]) #first term becomes sed_added + prev sed
        dS_f[i] = sed_added[i] - Hs_out[i]

    else:
        Hs_out[i] = sed_cap[i]
        dS_f[i] = sed_added[i] - Hs_out[i]

    S_f[i] = S_f[i-1] + dS_f[i] if (S_f[i-1] + dS_f[i]) > 0 else 0
    
#Add all numpy arrays to the Pandas dataframe
df['q_s'] = q_s
df_storage['n_t'] = n_t
df_storage['ks'] = k_s
df_storage['water_depth'] = H
df_storage['shear_stress'] = shear_stress
df_storage['hf'] = h_f
df_storage['f_s'] = f_s
df_storage['q_s'] = q_s
df_storage['qf1'] = q_f1
df_storage['qf2'] = q_f2
df_storage['dS_f'] = dS_f
df_storage['S_f'] = S_f
df_storage['S_s'] = S_s
df_storage['S_sc'] = S_sc
df_storage['S_sf'] = S_sf
df_storage['S_b'] = S_b
df_storage['S_bc'] = S_bc
df_storage['S_bf'] = S_bf
df_storage['Hs_out'] = Hs_out
df_storage['sed_added'] = sed_added
df_storage['sed_cap'] = sed_cap

#%%

plt.figure(figsize=(6,4))

_= df_storage.hf.plot()

plt.xlabel('Date')
plt.ylabel(r'$h_f$')
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\New_SSP.png', dpi=300)

#%%
plt.figure(figsize=(6,4))

_= df_storage.f_s.plot()

plt.xlabel('Date')
plt.ylabel(r'$f_s$')
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\New_SSP.png', dpi=300)

f_s.max()
#%%
#Resample to daily data again
df_day_sed = df.resample('D').sum().fillna(0)
df_day_sed['day'] = np.arange(0, len(df_day_sed), 1)

#Plot sediment transport rates over time
fig2, ax2 = plt.subplots(figsize=(7,5))
df_day_sed.plot(y='q_s', ax=ax2, color = 'peru', legend=False)
plt.xlabel('Date')
plt.ylabel(r'Sediment transport rate $(m/s)$')
plt.title('Sediment transport rates', fontweight='bold', fontsize=14)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\Sediment.png', dpi=300)
plt.show()
#%%
df4 = df_storage.resample('D').mean().fillna(method='ffill')
df4['day'] = np.arange(0, len(df4), 1)
df4['S_f_mm'] = df4.S_f*1000
df4['sed_cap_mm'] = df4.sed_cap/1e-3
df4['Hs_out_mm'] = df4.Hs_out/1e-3
#%%
fig13, ax13 = plt.subplots(figsize=(6,4))
df4.sed_cap_mm.plot(color = '#9e80c2', label='Transport capacity')
df4.Hs_out_mm.plot(color='#442766', label='Actual transport')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel(r'Sediment depth $(mm)$', fontsize=14, fontweight='bold')
#plt.ylim(0,8)

fig13.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax13.transAxes)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\CapacityVTransport_New.png', dpi=300)
plt.show()
#%%
fig3, ax3 = plt.subplots(figsize=(9,4.5))
df4.plot(y='S_f_mm', ax=ax3, color = 'mediumseagreen', legend=False)
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel(r'Fine sediment storage, $S_f$ $(mm)$',fontsize=14, fontweight='bold')
#plt.title('Fine sediment storage', fontweight='bold', fontsize=14)
#plt.ylim(0,1.4)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\FineStorage.png', dpi=300)
plt.show()
#%%
df4['sed_cap_mm'] = df4.sed_cap/1e-3

fig8, ax8 = plt.subplots(figsize=(7,3))
df4.plot(y='S_f_mm', ax=ax8, color = 'mediumseagreen', legend=False, label='Fine storage')
df4.plot(y='Hs_out_mm', ax=ax8, color = '#442766', legend=False, label='Actual transport', alpha=0.75)

ax8.set_xlabel('Date', fontsize=14, fontweight='bold')
ax8.set_ylabel(r'Sediment depth $(mm)$', fontsize=14, fontweight='bold')
#ax8.set_ylim(0, 1.4)

fig8.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax8.transAxes)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\CapacityVStorage.png', dpi=300)
plt.show()
#%%
fig10, ax10 = plt.subplots(figsize=(10,7))
df4.plot(y='qf1', ax=ax10, color = 'mediumturquoise', legend=False, label=r'$q_{f1}$')
df4.plot(y='qf2', ax=ax10, color = 'mediumvioletred', legend=False, label=r'$q_{f2}$')

ax10.set_ylabel(r'Sediment flux $(mm/s)$')
ax10.set_xlabel('Date')
fig10.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax10.transAxes)

plt.title('Sediment layer fluxes', fontweight='bold', fontsize=14)
plt.show()
#%%
df4['diff'] = df4.qf1-df4.qf2
fig11, ax11 = plt.subplots(figsize=(10,7))
df4.plot(y='diff', ax=ax11, color = 'mediumturquoise', legend=False)
ax11.set_ylabel(r'Flux difference, $(mm/s)$')
ax11.set_xlabel('Date')
plt.show()
#%%
df5 = df_storage.resample('D').mean().fillna(method='ffill')
#df5['hour'] = np.arange(0, len(df5), 1)
#%%
fig5, ax5 = plt.subplots(figsize=(6,4))

df5.plot(y='S_s', ax=ax5, color = '#532287', legend=False, label='Total surfacing')
df5.plot(y='S_sc', ax=ax5, color = '#b0077d', legend=False, label='Coarse surfacing')
df5.plot(y='S_sf', ax=ax5, color = '#027fcc', legend=False, label='Fine surfacing')

ax5.set_ylabel(r'Surfacing storage, $S_f$ $(m)$', fontweight='bold', fontsize=14)
#ax5.set_ylim(0, 0.25)
#ax5.set_title('Surfacing storage', fontweight='bold', fontsize=14)
fig5.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax5.transAxes)
plt.xlabel('Date', fontweight='bold', fontsize=14)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\SurfStorage_Ch10.png', dpi=300)
plt.show()
#%%
fig7, ax7 = plt.subplots(figsize=(6,4))
df5.plot(y='S_b', ax=ax7, color = '#12586b', legend=False, label='Total ballast')
df5.plot(y='S_bc', ax=ax7, color = '#099c49', legend=False, label='Coarse ballast')
df5.plot(y='S_bf', ax=ax7, color = '#2949e6', legend=False, label='Fine ballast')
#ax7.set_ylim(0, 0.7)
plt.xlabel('Date', fontweight='bold', fontsize=14)
plt.ylabel(r'Ballast storage, $S_b$ $(m)$', fontweight='bold', fontsize=14)
fig7.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax7.transAxes)
#plt.title('Ballast storage', fontweight='bold', fontsize=14)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\BalStorage_Ch10.png', dpi=300)
plt.show()
#%%
fig10, ax10 = plt.subplots(3, figsize=(9,7), sharex=True)
df4.plot(y='S_f_mm', ax=ax10[0], color = 'mediumseagreen', legend=False, label='TAF elevation')
df5.plot(y='S_s', ax=ax10[1], color = '#532287', legend=False, label='Surfacing elevation')
df5.plot(y='S_b', ax=ax10[2], color = '#12586b', legend=False, label='Ballast elevation')

plt.xlabel('Date', fontweight='bold', fontsize=14)
ax10[0].set_ylabel(r'$S_f$ $(mm)$', fontweight='bold', fontsize=14)
ax10[1].set_ylabel(r'$S_s$ $(m)$', fontweight='bold', fontsize=14)
ax10[2].set_ylabel(r'$S_b$ $(m)$', fontweight='bold', fontsize=14)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\100yrs_Ch10.png', dpi=300)
plt.show()
#%%
#Subset data by water year
yr_1 = df_storage.Hs_out['2018-10-01':'2019-09-30'].sum()
yr_2 = df_storage.Hs_out['2019-10-01':'2020-09-30'].sum()
yr_3 = df_storage.Hs_out['2020-10-01':'2021-09-30'].sum()
yr_4 = df_storage.Hs_out['2021-10-01':'2022-09-30'].sum()
yr_5 = df_storage.Hs_out['2022-10-01':'2023-09-30'].sum()
yr_6 = df_storage.Hs_out['2023-10-01':'2024-09-30'].sum()
yr_7 = df_storage.Hs_out['2024-10-01':'2025-09-30'].sum()
yr_8 = df_storage.Hs_out['2025-10-01':'2026-09-30'].sum()
yr_9 = df_storage.Hs_out['2026-10-01':'2027-09-30'].sum()
yr_10 = df_storage.Hs_out['2027-10-01':'2028-09-30'].sum()
yr_11 = df_storage.Hs_out['2028-10-01':'2029-09-30'].sum()
yr_12 = df_storage.Hs_out['2029-10-01':'2030-09-30'].sum()
yr_13 = df_storage.Hs_out['2030-10-01':'2031-09-30'].sum()
yr_14 = df_storage.Hs_out['2031-10-01':'2032-09-30'].sum()
yr_15 = df_storage.Hs_out['2032-10-01':'2033-09-30'].sum()
yr_16 = df_storage.Hs_out['2033-10-01':'2034-09-30'].sum()
yr_17 = df_storage.Hs_out['2034-10-01':'2035-09-30'].sum()
yr_18 = df_storage.Hs_out['2035-10-01':'2036-09-30'].sum()
yr_19 = df_storage.Hs_out['2036-10-01':'2037-09-30'].sum()
yr_20 = df_storage.Hs_out['2037-10-01':'2038-09-30'].sum()


#Multiply Hs_out
sed_area = np.multiply([yr_1, yr_2, yr_3, yr_4, yr_5, yr_6, yr_7, \
                        yr_8, yr_9, yr_10, yr_11, yr_12, yr_13, yr_14, \
                        yr_15, yr_16, yr_17, yr_18, yr_19, yr_20], L)
sed_load = np.multiply(sed_area, rho_s)
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, \
         2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038]
#%%
ticks = years
fig8, ax8 = plt.subplots(figsize=(6,4))
plt.bar(years, sed_load, color = '#1d4f54')
plt.xlabel('Water year', fontweight='bold', fontsize=14)
plt.ylabel(r'Mass per meter of road $(kg/m)$', fontweight='bold', fontsize=14)
#plt.title('Yearly sediment load per meter of road', fontweight='bold', fontsize=14)
plt.xticks(range(ticks[0],ticks[len(ticks)-1]+1), ticks, rotation=45)
plt.tight_layout()
#plt.savefig(r'C:\Users\Amanda\Desktop\AnnualYield_New.png', dpi=300)
plt.show()
#%%
sed_sum_m2 = df_storage.Hs_out.sum()
sed_sum_kg_m = sed_sum_m2*rho_s*L
round(sed_sum_kg_m)
#%%
s = (df_storage.S_s[0]-df_storage.S_s[len(df_storage)-1])
b = (df_storage.S_b[0]-df_storage.S_b[len(df_storage)-1])
f = (df_storage.S_f[0]-df_storage.S_f[len(df_storage)-1])

round((s+b+f)*rho_s*L)