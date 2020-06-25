"""
Author: Amanda Manaster
Date: 06/16/2020
Purpose: Lumped model of road prism forced using hourly weather station data  
         at North Fork Toutle; Lat: 46.37194, Lon: -122.57778
         Data pulled from Mesonet (https://developers.synopticdata.com/mesonet/)
"""
#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

#Read in .csv of pre-grouped storms
data_df = pd.read_csv('./rlm_output/groupedStorms_toutle.csv', index_col='date')
data_df.index = pd.to_datetime(data_df.index)

#Length of storm in # of hourly time steps
timeStep_Hr = data_df.groupby('totalNo')['totalNo'].count().to_numpy()
#Depth of storm in mm
groupedDepth = data_df.groupedDepth

deltaT = timeStep_Hr #time step
totalT = np.cumsum(deltaT) #total time
truck_pass = [] #initialize empty list for truck passes

np.random.seed(1) #Use seed to ensure consistent results with each run

###CHECK THIS!!!!!!!!!
for i, time in enumerate(timeStep_Hr):
    truck = 0
    if time/24 >= 1:
        day = int(time/24)
        frac_day = time/24 - int(time/24)
        for num in range(day):
            truck += np.random.poisson(10,1).item()
        truck += round(np.random.poisson(10,1).item()*frac_day)
    else:
        frac_day = time/24 - int(time/24)
        truck = round(np.random.poisson(10,1).item()*frac_day)
    truck_pass.append(truck) #number of truck passes

#Create new dataframe that's grouped into storm/interstorm time periods
storms_df = pd.DataFrame()
storms_df['time'] = totalT
storms_df['deltaT'] = deltaT
storms_df['day'] = np.divide(totalT,24).astype('int64')
storms_df['truck_pass'] = truck_pass
storms_df['stormNo'] = data_df.groupby('stormNo')['stormNo'].mean()
storms_df['intensity'] = data_df.groupby('stormNo')['intensity_mmhr'].mean() 
# ^^normal average intensity; later we get weighted averaged intensity

day0 = data_df.index[0]
storms_df.set_index(pd.DatetimeIndex([day0+datetime.timedelta(hours=time) 
    for time in storms_df.time]), inplace=True)

#Define physical constants
# L = representative segment of road, m
# S = m/m
# tau_c = N/m^2; value from https://pubs.usgs.gov/sir/2008/5093/table7.html 
#     =====> 0.0091 mm is avg
L, rho_w, rho_s, g, S, tau_c, d50, d95 = [4.57, 1000, 2650, 
                                          9.81, 0.03, 0.05,
                                          9.1e-6, 0.0275]
#Define layer constants
# h_s = depth of surfacing
# f_sf, f_sc = fractions of fine/coarse material in ballast
# h_b = depth of surfacing
# f_bf, f_bc = fractions of fine/coarse material in ballast
h_s, f_sf, f_sc = [0.23, 0.275, 0.725]
h_b, f_bf, f_br = [2, 0.20, 0.80]

#The following four constants can be adjusted based on observations
# k_as = crushing constant for surfacing, m/truck pass
# k_ab = crushing constant for ballast, m/truck pass
# u_ps = pumping constant for surfacing, m/truck pass
#   (2.14e-5m^3/4.57 m^2)  
#   6 tires * 0.225 m width * 0.005 m length * 3.175e-3 m treads
# u_pb = pumping constant for ballast, m/truck pass
# e = fraction of coarse material, -
k_as, k_ab, u_ps, u_pb, e = [1.37e-6, 1.00e-6, 
                             4.69e-6, 2.345e-6, 
                             0.725]

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


#Step 1!
#Initialize numpy arrays for calculations
dS_f = np.zeros(len(storms_df))
S_f = np.zeros(len(storms_df))
S_s = np.zeros(len(storms_df))
S_sc = np.zeros(len(storms_df))
S_sf = np.zeros(len(storms_df))
S_b = np.zeros(len(storms_df))
S_bc = np.zeros(len(storms_df))
S_bf = np.zeros(len(storms_df))
Hs_out = np.zeros(len(storms_df))
S_f_init = np.zeros(len(storms_df))
test = np.zeros(len(storms_df))

q = np.zeros(len(int_tip_df))
q_avg = np.zeros(len(int_tip_df))

q_storm = np.zeros(len(storms_df))
f_s = np.zeros(len(storms_df))
n_f = np.zeros(len(storms_df))
n_c = np.zeros(len(storms_df))
n_t = np.zeros(len(storms_df))
q_s = np.zeros(len(storms_df))
q_ref = np.zeros(len(storms_df))
water_depth = np.zeros(len(storms_df))
tau = np.zeros(len(storms_df))
tau_e = np.zeros(len(storms_df))

n_tp = storms_df.truck_pass.to_numpy()
t = storms_df.deltaT.to_numpy()
t_storm = int_tip_df.groupby('stormNo')['storm_dur'].mean().to_numpy()

rainfall = int_tip_df.intensity_mmhr.to_numpy()
frac = int_tip_df.frac.to_numpy()
stormNo = int_tip_df.stormNo.to_numpy()

q_f1 = np.zeros(len(storms_df))
q_f2 = np.zeros(len(storms_df))
q_as = np.zeros(len(storms_df))
q_ab = np.zeros(len(storms_df))
sed_added = np.zeros(len(storms_df))
sed_cap = np.zeros(len(storms_df))
value = np.zeros(len(storms_df))
ref_trans = np.zeros(len(storms_df))

#Initial conditions for fines, surfacing, ballast
S_f_init[0] = 0.0275
S_f[0] = 0.0275
S_s[0] = h_s*(f_sf + f_sc)
S_sc[0] = h_s*(f_sc)
S_sf[0] = h_s*(f_sf)
S_b[0] = h_b*(f_bf + f_br)
S_bc[0] = h_b*(f_br)
S_bf[0] = h_b*(f_bf)

for j, storm in enumerate(storms_df.stormNo):
    if j == 0:
        continue
    else: ####CHECK THIS!!!!!!!!!
        q_f1[j] = u_ps*(S_sf[j-1]/S_s[j-1])*n_tp[j]/(t[j]*3600)
        q_f2[j] = u_pb*(S_bf[j-1]/S_b[j-1])*n_tp[j]/(t[j]*3600)
        q_as[j] = k_as*(S_sc[j-1]/S_s[j-1])*n_tp[j]/(t[j]*3600)
        q_ab[j] = k_ab*(S_bc[j-1]/S_b[j-1])*n_tp[j]/(t[j]*3600)
        S_bc[j] = S_bc[j-1] - q_ab[j]*(t[j]*3600)
        S_sc[j] = S_sc[j-1] - q_as[j]*(t[j]*3600)
        S_bf[j] = S_bf[j-1] + q_ab[j]*(t[j]*3600) - q_f2[j]*(t[j]*3600)
        S_sf[j] = S_sf[j-1] + q_as[j]*(t[j]*3600) - q_f1[j]*(t[j]*3600) +\
                  q_f2[j]*(t[j]*3600)
        S_s[j] = S_sc[j] + S_sf[j]
        S_b[j] = S_bc[j] + S_bf[j]

    if d95 >= S_f[j-1]:
        # Let's try using an Sf_init variable because when sediment is added to
        # TAF, that doesn't change the final S_f value, but it does affect how
        # the sediment is transported!
        sed_added[j] = (q_f1[j]*(t[j]*3600.))/(1-e)
        S_f_init[j] = S_f[j-1] + sed_added[j]
    else:
        sed_added[j] = q_f1[j]*(t[j]*3600.)
        S_f_init[j] = S_f[j-1] + sed_added[j]

    for k, val in enumerate(stormNo):
        q[k] = rainfall[k]*2.77778e-7*L 
        q_avg[k] = q[k]*frac[k]
    
        if val == storm:
            q_storm[j] += q_avg[k]
    
    if q_storm[j] > 0:
        n_f[j] = 0.0026*q_storm[j]**(-0.274)
        n_c[j] = 0.08*q_storm[j]**(-0.153)
    else:
        n_f[j] = n_f[j-1]
        n_c[j] = n_c[j-1]

    if S_f_init[j] <= d95:
        n_t[j] = n_c[j] + (S_f_init[j]/d95)*(n_f[j]-n_c[j])
        f_s[j] = (n_f[j]/n_t[j])**(1.5)*(S_f_init[j]/d95)
    else: 
        n_t[j] = n_f[j]
        f_s[j] = (n_f[j]/n_t[j])**(1.5)

    #Calculate water depth assuming uniform overland flow
    water_depth[j] = ((n_t[j]*q_storm[j])/(S**(1/2)))**(3/5)

    tau[j] = rho_w*g*water_depth[j]*S
    tau_e[j] = tau[j]*f_s[j]
    
    #Calculate sediment transport rate
    if (tau_e[j]-tau_c) >= 0:
        q_s[j] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*\
                 (tau_e[j]-tau_c)**(2.457))
    else:
        q_s[j] = 0
    
    #Calculate reference transport 
    if (tau[j]-tau_c) >=0:
        q_ref[j] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*\
                       (tau[j]-tau_c)**(2.457))
    else:
        q_ref[j] = 0

    sed_cap[j] = q_s[j]*t_storm[j]*3600/L
    ref_trans[j] = q_ref[j]*t_storm[j]*3600/L

    #Create a condition column based on sediment transport capacity vs sediment supply
    value[j] = (sed_added[j]-sed_cap[j]) #*******Are we considering ref transport or actual transport here??????
        
    if value[j] < 0:
        Hs_out[j] = np.minimum(sed_added[j]+S_f[j-1], sed_cap[j])
    else:
        Hs_out[j] = sed_cap[j]
    
    dS_f[j] = sed_added[j] - Hs_out[j]
    S_f[j] = S_f[j-1] + dS_f[j] #if (S_f_init[j] - Hs_out[j]) > 0 else 0

#Step 2!
#Add all numpy arrays to the Pandas dataframe
storms_df['qf1'] = q_f1
storms_df['qf2'] = q_f2
storms_df['S_f'] = S_f*1000
storms_df['S_s'] = S_s
storms_df['S_sc'] = S_sc
storms_df['S_sf'] = S_sf
storms_df['S_b'] = S_b
storms_df['S_bc'] = S_bc
storms_df['S_bf'] = S_bf
storms_df['Hs_out'] = Hs_out*1000
storms_df['sed_added'] = sed_added
storms_df['sed_cap'] = sed_cap*1000
storms_df['ref_trans'] = ref_trans*1000
storms_df['val'] = value
storms_df['water_depth'] = water_depth
storms_df['tau'] = tau
storms_df['tau_e'] = tau_e
storms_df['n_t'] = n_t
storms_df['f_s'] = f_s
storms_df['qs'] = q_s
storms_df['q_storm'] = q_storm

int_tip_df['q'] = q
int_tip_df['q_avg'] = q_avg


plt.close('all')

#Plot f_s over time
# fig1, ax1 = plt.subplots(figsize=(6,4))
# storms_df.f_s.plot(ax=ax1)
# plt.xlabel('Date')
# plt.ylabel(r'$f_s$')
# plt.tight_layout()
# plt.show()

#Plot sediment transport rates over time
# fig2, ax2 = plt.subplots(figsize=(7,5))
# int_tip_df.plot(y='qs', ax=ax2, color = 'peru', legend=False)
# plt.xlabel('Date')
# plt.ylabel(r'Sediment transport rate $(m/s)$')
# plt.title('Sediment transport rates', fontweight='bold', fontsize=14)
# plt.tight_layout()
# plt.show()

#Plot sediment transport capacity and actual transport over time
fig3, ax3 = plt.subplots(figsize=(6,4))
storms_df.ref_trans.plot(color = '#9e80c2', label='Reference transport capacity')
storms_df.Hs_out.plot(linestyle='--', color='#442766', label='Actual transport')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel(r'Sediment depth $(mm)$', fontsize=14, fontweight='bold')
fig3.legend(loc="upper right", bbox_to_anchor=(1,1), 
    bbox_transform=ax3.transAxes)
plt.tight_layout()
plt.show()

#Plot fine sediment storage over time
fig4, ax4 = plt.subplots(figsize=(9,4.5))
storms_df.plot(y='S_f', ax=ax4, color = 'mediumseagreen', legend=False)
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel(r'Fine sediment storage, $S_f$ $(mm)$',fontsize=14, 
    fontweight='bold')
plt.title('Fine sediment storage', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

#Plot fine sediment storage and actual transport over time
fig5, ax5 = plt.subplots(figsize=(7,3))
storms_df.plot(y='S_f', ax=ax5, color = 'mediumseagreen', legend=False, 
    label='Fine storage')
storms_df.plot(y='Hs_out', ax=ax5, color = '#442766', legend=False, 
    label='Actual transport', alpha=0.75)
ax5.set_xlabel('Date', fontsize=14, fontweight='bold')
ax5.set_ylabel(r'Sediment depth $(mm)$', fontsize=14, fontweight='bold')
fig5.legend(loc="upper right", bbox_to_anchor=(1,1), 
    bbox_transform=ax5.transAxes)
plt.tight_layout()
plt.show()

#Plot transportative fluxes between layers
# fig6, ax6 = plt.subplots(figsize=(10,7))
# storms_df.plot(y='qf1', ax=ax6, color = 'mediumturquoise', 
#     legend=False, label=r'$q_{f1}$')
# storms_df.plot(y='qf2', ax=ax6, color = 'mediumvioletred', 
#     legend=False, label=r'$q_{f2}$')
# ax6.set_ylabel(r'Sediment flux $(mm/s)$')
# ax6.set_xlabel('Date')
# fig6.legend(loc="upper right", bbox_to_anchor=(1,1), 
#     bbox_transform=ax6.transAxes)
# plt.title('Sediment layer fluxes', fontweight='bold', fontsize=14)
# #plt.show()

#Plot surfacing storage over time
fig7, ax7 = plt.subplots(figsize=(6,4))
storms_df.plot(y='S_s', ax=ax7, color = '#532287', legend=False, 
    label='Total surfacing')
storms_df.plot(y='S_sc', ax=ax7, color = '#b0077d', legend=False, 
    label='Coarse surfacing')
storms_df.plot(y='S_sf', ax=ax7, color = '#027fcc', legend=False, 
    label='Fine surfacing')
ax7.set_ylabel(r'Surfacing storage, $S_f$ $(m)$', fontweight='bold', 
    fontsize=14)
ax7.set_title('Surfacing storage', fontweight='bold', fontsize=14)
fig7.legend(loc="upper right", bbox_to_anchor=(1,1), 
    bbox_transform=ax7.transAxes)
plt.xlabel('Date', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

#Plot ballast storage over time
fig8, ax8 = plt.subplots(figsize=(6,4))
storms_df.plot(y='S_b', ax=ax8, color = '#12586b', legend=False, 
    label='Total ballast')
storms_df.plot(y='S_bc', ax=ax8, color = '#099c49', legend=False, 
    label='Coarse ballast')
storms_df.plot(y='S_bf', ax=ax8, color = '#2949e6', legend=False, 
    label='Fine ballast')
plt.xlabel('Date', fontweight='bold', fontsize=14)
plt.ylabel(r'Ballast storage, $S_b$ $(m)$', fontweight='bold', 
    fontsize=14)
fig8.legend(loc="upper right", bbox_to_anchor=(1,1), 
    bbox_transform=ax8.transAxes)
plt.title('Ballast storage', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# fig9, ax9 = plt.subplots(3, figsize=(9,7), sharex=True)
# storms_df.plot(y='S_f', ax=ax9[0], color = 'mediumseagreen', 
#     legend=False, label='TAF elevation')
# storms_df.plot(y='S_s', ax=ax9[1], color = '#532287', legend=False, 
#     label='Surfacing elevation')
# storms_df.plot(y='S_b', ax=ax9[2], color = '#12586b', legend=False, 
#     label='Ballast elevation')
# plt.xlabel('Date', fontweight='bold', fontsize=14)
# ax9[0].set_ylabel(r'$S_f$ $(mm)$', fontweight='bold', fontsize=14)
# ax9[1].set_ylabel(r'$S_s$ $(m)$', fontweight='bold', fontsize=14)
# ax9[2].set_ylabel(r'$S_b$ $(m)$', fontweight='bold', fontsize=14)
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

sed_sum_m = storms_df.sed_added.sum()-(storms_df.Hs_out.sum()/1000)
sed_sum_kg_m = sed_sum_m*rho_s*L
f = ((storms_df.S_f[len(storms_df)-1]-storms_df.S_f[0])/1000)*rho_s*L

if round(f) == round(sed_sum_kg_m):
    print('\nThe mass balance is fine.')
else:
    print('\nThe mass balance is off.')

total_out_kg = (storms_df.Hs_out.sum()/1000)*rho_s*L
print("\nTotal amount of sediment transported:", round(total_out_kg), "kg/m")

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