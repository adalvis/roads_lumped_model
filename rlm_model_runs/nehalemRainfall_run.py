"""
Author: Amanda Manaster
Date: 04/17/2020
Purpose: Lumped model of road prism forced using 15 minute tipping bucket data
         from Nehalem, OR.
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

data_df = pd.read_csv('./rlm_output/groupedStorms.csv', index_col='date')
data_df.index = pd.to_datetime(data_df.index)

test = data_df.groupby(['totalNo','depth', 'intensity'])['tips'].count() # len=426

# length of storm in # of quarter hour time steps
timeStep_qtrHr = data_df.groupby('totalNo')['totalNo'].count().to_numpy()

# length of storm in # of hourly time steps
timeStep_Hr = timeStep_qtrHr/4

groupedDepth = data_df.groupby('stormNo')['depth'].sum().to_numpy()

deltaT = timeStep_Hr
totalT = np.cumsum(deltaT)
truck_pass = []

np.random.seed(1) #Use seed to ensure consistent results with each run

for i, time in enumerate(timeStep_Hr):
    truck = 0
    if time/24 >= 1:
        day = int(time/24)
        frac_day = time/24 - int(time/24)
        for num in range(day):
            truck += np.random.randint(0,10)
        truck += round(np.random.randint(0,10)*frac_day)
    else:
        frac_day = time/24 - int(time/24)
        truck = round(np.random.randint(0,10)*frac_day)
    truck_pass.append(truck) #number of truck passes

#Create new dataframe that's grouped into storm/interstorm time periods
storms_df = pd.DataFrame()
storms_df['time'] = totalT
storms_df['deltaT'] = deltaT
storms_df['day'] = np.divide(totalT,24).astype('int64')
storms_df['truck_pass'] = truck_pass
storms_df['stormNo'] = data_df.groupby('stormNo')['stormNo'].mean()
storms_df['intensity'] = data_df.groupby('intensity')['intensity'].mean()

day0 = data_df.index[0]
storms_df.set_index(pd.DatetimeIndex([day0+datetime.timedelta(hours=time) 
    for time in storms_df.time]), inplace=True)

#Define physical constants
# L = representative segment of road, m
# S = m/m; 8% long slope, 2% lat slope
# tau_c = N/m^2; assuming d50 is approx. 0.0580 mm; 
#         value from https://pubs.usgs.gov/sir/2008/5093/table7.html
# n_f = 0.03 #0.0475*(d50)**(1/6); Manning's n of fines in TAF
L, rho_w, rho_s, g, S, tau_c, d50, d95, n_f = [4.57, 1000, 2650, 
                                               9.81, 0.0825, 0.04,
                                               6.25e-5, 0.0375, 0.03]
#Define layer constants
# h_s = depth of surfacing
# f_sf, f_sc = fractions of fine/coarse material in ballast
# h_b = depth of surfacing
# f_bf, f_bc = fractions of fine/coarse material in ballast
h_s, f_sf, f_sc = [0.23, 0.275, 0.725]
h_b, f_bf, f_br = [2, 0.20, 0.80]

#The following four constants can be adjusted based on observations
# kas = crushing constant for surfacing, m/truck pass
# kab = crushing constant for ballast, m/truck pass
# u_ps = pumping constant for surfacing, m/truck pass
#   (2.14e-5m^3/4.57 m^2)  
#   6 tires * 0.225 m width * 0.005 m length * 3.175e-3 m treads
# u_pb = pumping constant for ballast, m/truck pass
# e = fraction of coarse material, -
k_as, k_ab, u_ps, u_pb, e = [1.37e-7, 1.00e-7, 4.69e-7, 2.345e-7, 0.725]

for j, storm in enumerate(storms_df.stormNo):
    ಠ_ಠ = test.loc[storm,:,:]/test.loc[storm,:,:].sum()

# #Step 1!
# #Initialize numpy arrays for calculations
# dS_f = np.zeros(len(storms_df))
# S_f = np.zeros(len(storms_df))
# S_s = np.zeros(len(storms_df))
# S_sc = np.zeros(len(storms_df))
# S_sf = np.zeros(len(storms_df))
# S_b = np.zeros(len(storms_df))
# S_bc = np.zeros(len(storms_df))
# S_bf = np.zeros(len(storms_df))
# Hs_out = np.zeros(len(storms_df))
# q_s = np.zeros(len(storms_df))
# k_s = np.zeros(len(storms_df))
# H = np.zeros(len(storms_df))
# tau = np.zeros(len(storms_df))
# shear_stress = np.zeros(len(storms_df))
# f_s = np.zeros(len(storms_df))
# n_c = np.zeros(len(storms_df))
# n_t = np.zeros(len(storms_df))
# S_f_init = np.zeros(len(storms_df))

# n_tp = storms_df.truck_pass.to_numpy()
# t = storms_df.deltaT.to_numpy()
# t_storm = storms_df.storm_length.to_numpy()
# rainfall = storms_df.rainfall_rate.to_numpy()

# q_f1 = np.zeros(len(storms_df))
# q_f2 = np.zeros(len(storms_df))
# q_as = np.zeros(len(storms_df))
# q_ab = np.zeros(len(storms_df))
# sed_added = np.zeros(len(storms_df))
# sed_cap = np.zeros(len(storms_df))
# value = np.zeros(len(storms_df))

# #Initial conditions for fines, surfacing, ballast
# S_f_init[0] = 0
# n_c[0] = 0.035 #0.0475*(d95-S_f_init[0])**(1/6)
# n_t[0] = n_f+n_c[0]
# f_s[0] = (n_f/n_t[0])**(1.5)
# S_f[0] = 0
# S_s[0] = h_s*(f_sf + f_sc)
# S_sc[0] = h_s*(f_sc)
# S_sf[0] = h_s*(f_sf)
# S_b[0] = h_b*(f_bf + f_br)
# S_bc[0] = h_b*(f_br)
# S_bf[0] = h_b*(f_bf)

# #Step 2!
# for i in range(1, len(storms_df)):
#     q_f1[i] = u_p*(S_sf[i-1]/S_s[i-1])*n_tp[i]/(t[i]*3600)
#     q_f2[i] = u_f*(S_bf[i-1]/S_b[i-1])*n_tp[i]/(t[i]*3600)
#     q_as[i] = kas*(S_sc[i-1]/S_s[i-1])*n_tp[i]/(t[i]*3600)
#     q_ab[i] = kab*(S_bc[i-1]/S_b[i-1])*n_tp[i]/(t[i]*3600)
    
#     S_bc[i] = S_bc[i-1] - q_ab[i]*(t[i]*3600)
#     S_sc[i] = S_sc[i-1] - q_as[i]*(t[i]*3600)
    
#     S_bf[i] = S_bf[i-1] + q_ab[i]*(t[i]*3600) - q_f2[i]*(t[i]*3600)
#     S_sf[i] = S_sf[i-1] + q_as[i]*(t[i]*3600) - q_f1[i]*(t[i]*3600) +\
#               q_f2[i]*(t[i]*3600)
        
#     S_s[i] = S_sc[i] + S_sf[i]
#     S_b[i] = S_bc[i] + S_bf[i]
    
#     if d95 >= S_f[i-1]:
#         #Let's try using an Sf_init variable because when sediment is added to
#         # TAF, that doesn't change the final S_f value, but it does affect how
#         # the sediment is transported!
#         sed_added[i] = (q_f1[i]*(t[i]*3600.))/(1-e)
#         S_f_init[i] = S_f[i-1] + sed_added[i]
#     else:
#         sed_added[i] = q_f1[i]*(t[i]*3600.)
#         S_f_init[i] = S_f[i-1] + sed_added[i]
    
#     if d95 > S_f_init[i]:
#         k_s[i] = d95 - S_f_init[i]
#         n_c[i] = 0.035 #0.0475*(k_s[i])**(1/6)
#     else:
#         n_c[i] = 0
    
#     n_t[i] = 0.06
    
#     f_s[i] = (n_f/n_t[i])**(1.5)
        
#     #Calculate water depth assuming uniform overland flow
#     H[i] = ((n_t[i]*(rainfall[i]*2.77778e-7)*L)/(S**(1/2)))**(3/5)
    
#     tau[i] = rho_w*g*H[i]*S
    
#     #Calculate shear stress
#     shear_stress[i] = tau[i]*f_s[i]
    
#     #Calculate sediment transport rate
#     if (shear_stress[i]-tau_c) >= 0:
#         q_s[i] = ((10**(-4.348))/(rho_s*((d50)**(0.811))))*\
#                  (shear_stress[i]-tau_c)**(2.457)/L
#     else:
#         q_s[i] = 0

#     #Create a condition column based on sediment transport capacity vs sediment supply     
#     sed_cap[i] = q_s[i]*(t_storm[i]*3600.)
#     value[i] = (sed_added[i]-sed_cap[i])
        
#     if value[i] < 0:
#         Hs_out[i] = np.minimum(sed_added[i]+S_f[i-1], sed_cap[i])
#         dS_f[i] = sed_added[i] - Hs_out[i]

#     else:
#         Hs_out[i] = sed_cap[i]
#         dS_f[i] = sed_added[i] - Hs_out[i]

#     S_f[i] = S_f[i-1] + dS_f[i] #if (S_f_init[i] - Hs_out[i]) > 0 else 0
    
# #Add all numpy arrays to the Pandas dataframe
# storms_df['q_s'] = q_s
# storms_df['n_t'] = n_t
# storms_df['ks'] = k_s
# storms_df['water_depth'] = H
# storms_df['shear_stress'] = shear_stress
# storms_df['S_f_init'] = S_f_init
# storms_df['f_s'] = f_s
# storms_df['q_s'] = q_s
# storms_df['qf1'] = q_f1
# storms_df['qf2'] = q_f2
# storms_df['dS_f'] = dS_f
# storms_df['S_f'] = S_f
# storms_df['S_s'] = S_s
# storms_df['S_sc'] = S_sc
# storms_df['S_sf'] = S_sf
# storms_df['S_b'] = S_b
# storms_df['S_bc'] = S_bc
# storms_df['S_bf'] = S_bf
# storms_df['Hs_out'] = Hs_out
# storms_df['sed_added'] = sed_added
# storms_df['sed_cap'] = sed_cap
# storms_df['val'] = value

# plt.figure(figsize=(6,4))
# storms_df.plot(x= 'S_f_init', y = 'f_s')

# plt.figure(figsize=(6,4))
# storms_df.f_s.plot()
# plt.xlabel('Date')
# plt.ylabel(r'$f_s$')
# plt.tight_layout()
# plt.show()

# f_s.max()

#Resample to daily data again
# df_day_sed = storms_df.resample('D').sum().fillna(0)
# df_day_sed['day'] = np.arange(0, len(df_day_sed), 1)

#Plot sediment transport rates over time
# fig2, ax2 = plt.subplots(figsize=(7,5))
# df_day_sed.plot(y='q_s', ax=ax2, color = 'peru', legend=False)
# plt.xlabel('Date')
# plt.ylabel(r'Sediment transport rate $(m/s)$')
# plt.title('Sediment transport rates', fontweight='bold', fontsize=14)
# plt.tight_layout()
# plt.show()

# df4 = storms_df.resample('D').mean().fillna(method='ffill')
# df4['day'] = np.arange(0, len(df4), 1)
# df4['S_f_mm'] = df4.S_f*1000
# df4['sed_cap_mm'] = df4.sed_cap/1e-3
# df4['Hs_out_mm'] = df4.Hs_out/1e-3

# fig13, ax13 = plt.subplots(figsize=(6,4))
# df4.sed_cap_mm.plot(color = '#9e80c2', label='Transport capacity')
# df4.Hs_out_mm.plot(color='#442766', label='Actual transport')
# plt.xlabel('Date', fontsize=14, fontweight='bold')
# plt.ylabel(r'Sediment depth $(mm)$', fontsize=14, fontweight='bold')
# plt.ylim(0,8)
# fig13.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax13.transAxes)
# plt.tight_layout()
# plt.show()

# fig3, ax3 = plt.subplots(figsize=(9,4.5))
# df4.plot(y='S_f_mm', ax=ax3, color = 'mediumseagreen', legend=False)
# plt.xlabel('Date', fontsize=14, fontweight='bold')
# plt.ylabel(r'Fine sediment storage, $S_f$ $(mm)$',fontsize=14, fontweight='bold')
# plt.title('Fine sediment storage', fontweight='bold', fontsize=14)
# plt.ylim(0,1.4)
# plt.tight_layout()
# plt.show()

# df4['sed_cap_mm'] = df4.sed_cap/1e-3
# fig8, ax8 = plt.subplots(figsize=(7,3))
# df4.plot(y='S_f_mm', ax=ax8, color = 'mediumseagreen', legend=False, label='Fine storage')
# df4.plot(y='Hs_out_mm', ax=ax8, color = '#442766', legend=False, label='Actual transport', alpha=0.75)
# ax8.set_xlabel('Date', fontsize=14, fontweight='bold')
# ax8.set_ylabel(r'Sediment depth $(mm)$', fontsize=14, fontweight='bold')
# ax8.set_ylim(0, 1.4)
# fig8.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax8.transAxes)
# plt.tight_layout()
# plt.show()

# fig10, ax10 = plt.subplots(figsize=(10,7))
# df4.plot(y='qf1', ax=ax10, color = 'mediumturquoise', legend=False, label=r'$q_{f1}$')
# df4.plot(y='qf2', ax=ax10, color = 'mediumvioletred', legend=False, label=r'$q_{f2}$')
# ax10.set_ylabel(r'Sediment flux $(mm/s)$')
# ax10.set_xlabel('Date')
# fig10.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax10.transAxes)
# plt.title('Sediment layer fluxes', fontweight='bold', fontsize=14)
# plt.show()

# df4['diff'] = df4.qf1-df4.qf2
# fig11, ax11 = plt.subplots(figsize=(10,7))
# df4.plot(y='diff', ax=ax11, color = 'mediumturquoise', legend=False)
# ax11.set_ylabel(r'Flux difference, $(mm/s)$')
# ax11.set_xlabel('Date')
# plt.show()

# df5 = storms_df.resample('D').mean().fillna(method='ffill')
# #df5['hour'] = np.arange(0, len(df5), 1)

# fig5, ax5 = plt.subplots(figsize=(6,4))
# df5.plot(y='S_s', ax=ax5, color = '#532287', legend=False, label='Total surfacing')
# df5.plot(y='S_sc', ax=ax5, color = '#b0077d', legend=False, label='Coarse surfacing')
# df5.plot(y='S_sf', ax=ax5, color = '#027fcc', legend=False, label='Fine surfacing')
# ax5.set_ylabel(r'Surfacing storage, $S_f$ $(m)$', fontweight='bold', fontsize=14)
# ax5.set_ylim(0, 0.25)
# ax5.set_title('Surfacing storage', fontweight='bold', fontsize=14)
# fig5.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax5.transAxes)
# plt.xlabel('Date', fontweight='bold', fontsize=14)
# plt.tight_layout()
# plt.show()

# fig7, ax7 = plt.subplots(figsize=(6,4))
# df5.plot(y='S_b', ax=ax7, color = '#12586b', legend=False, label='Total ballast')
# df5.plot(y='S_bc', ax=ax7, color = '#099c49', legend=False, label='Coarse ballast')
# df5.plot(y='S_bf', ax=ax7, color = '#2949e6', legend=False, label='Fine ballast')
# ax7.set_ylim(0, 0.7)
# plt.xlabel('Date', fontweight='bold', fontsize=14)
# plt.ylabel(r'Ballast storage, $S_b$ $(m)$', fontweight='bold', fontsize=14)
# fig7.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax7.transAxes)
# plt.title('Ballast storage', fontweight='bold', fontsize=14)
# plt.tight_layout()
# plt.show()

# fig10, ax10 = plt.subplots(3, figsize=(9,7), sharex=True)
# df4.plot(y='S_f_mm', ax=ax10[0], color = 'mediumseagreen', legend=False, label='TAF elevation')
# df5.plot(y='S_s', ax=ax10[1], color = '#532287', legend=False, label='Surfacing elevation')
# df5.plot(y='S_b', ax=ax10[2], color = '#12586b', legend=False, label='Ballast elevation')

# plt.xlabel('Date', fontweight='bold', fontsize=14)
# ax10[0].set_ylabel(r'$S_f$ $(mm)$', fontweight='bold', fontsize=14)
# ax10[1].set_ylabel(r'$S_s$ $(m)$', fontweight='bold', fontsize=14)
# ax10[2].set_ylabel(r'$S_b$ $(m)$', fontweight='bold', fontsize=14)
# plt.tight_layout()
# plt.show()

# #Subset data by water year
# # yr_1 = storms_df.Hs_out['2018-10-01':'2019-09-30'].sum()
# # yr_2 = storms_df.Hs_out['2019-10-01':'2020-09-30'].sum()
# # yr_3 = storms_df.Hs_out['2020-10-01':'2021-09-30'].sum()
# # yr_4 = storms_df.Hs_out['2021-10-01':'2022-09-30'].sum()
# # yr_5 = storms_df.Hs_out['2022-10-01':'2023-09-30'].sum()
# # yr_6 = storms_df.Hs_out['2023-10-01':'2024-09-30'].sum()
# # yr_7 = storms_df.Hs_out['2024-10-01':'2025-09-30'].sum()
# # yr_8 = storms_df.Hs_out['2025-10-01':'2026-09-30'].sum()
# # yr_9 = storms_df.Hs_out['2026-10-01':'2027-09-30'].sum()
# # yr_10 = storms_df.Hs_out['2027-10-01':'2028-09-30'].sum()

# #Multiply Hs_out
# # sed_area = np.multiply([yr_1, yr_2, yr_3, yr_4, yr_5, yr_6, yr_7, \
# #                         yr_8, yr_9, yr_10], L)
# # sed_load = np.multiply(sed_area, rho_s)
# # years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]

# # ticks = years
# # fig8, ax8 = plt.subplots(figsize=(6,4))
# # plt.bar(years, sed_load, color = '#1d4f54')
# # plt.xlabel('Water year', fontweight='bold', fontsize=14)
# # plt.ylabel(r'Mass per meter of road $(kg/m)$', fontweight='bold', fontsize=14)
# # plt.title('Yearly sediment load per meter of road', fontweight='bold', fontsize=14)
# # plt.xticks(range(ticks[0],ticks[len(ticks)-1]+1), ticks, rotation=45)
# # plt.tight_layout()
# # plt.show()

# sed_sum_m2 = storms_df.Hs_out.sum()
# sed_sum_kg_m = sed_sum_m2*rho_s*L
# round(sed_sum_kg_m)
# s = (storms_df.S_s[0]-storms_df.S_s[len(storms_df)-1])
# b = (storms_df.S_b[0]-storms_df.S_b[len(storms_df)-1])
# f = (storms_df.S_f[0]-storms_df.S_f[len(storms_df)-1])
# print(round((s+b+f)*rho_s*L))


#Takes forever to run, hence down here.
# # ticklabels = [item.strftime('%Y') for item in df_day.index[::366*2]]

# # fig, ax = plt.subplots(figsize=(13,5))
# # df_day.plot(y='truck_pass', ax=ax, color = '#8a0c80', legend=False, label='Truck passes', 
# #             kind='bar', width=7)
# # ax.set_xlabel('Date', fontsize=14, fontweight='bold')
# # ax.set_ylabel('Truck passes', fontsize=14, fontweight='bold')
# # ax.grid(False)

# # ax1 = ax.twinx()
# # df_day.plot(y='storm_depth', ax=ax1, color='#0c3c8a', legend=False, label='Storm depth', kind='bar', width=7)
# # ax1.set_ylabel(r'Storm depth $(mm)$', fontsize=14, fontweight='bold')
# # ax1.invert_yaxis()
# # ax1.grid(False)

# # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
# # ax.set_xticks(np.arange(0,366*2*len(ticklabels),366*2))
# # ax.set_xticklabels(ticklabels, rotation=45)
# # plt.tight_layout()
# # #plt.savefig(r'C:\Users\Amanda\Desktop\Rainfall_Truck.png', dpi=300)
# # plt.show()