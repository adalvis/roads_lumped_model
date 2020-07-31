#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

def model_run(data_df, k_cs, k_cb, u_ps, u_pb, e):
    #Length of storm in # of hourly time steps
    timeStep_Hr = data_df.groupby('totalNo')['totalNo'].count().to_numpy()
    #Depth of storm in mm
    groupedDepth = data_df.groupedDepth

    deltaT = timeStep_Hr #time step
    totalT = np.cumsum(deltaT) #total time
    truck_pass = [] #initialize empty list for truck passes

    np.random.seed(2) #Use seed to ensure consistent results with each run

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
    storms_df['q_mean'] = storms_df.intensity*2.77778e-7*4.57

    day0 = data_df.index[0]
    storms_df.set_index(pd.DatetimeIndex([day0+datetime.timedelta(hours=time) 
        for time in storms_df.time]), inplace=True)

    #===========================DEFINE PHYSICAL CONSTANTS===========================
    # L = representative segment of road, m
    # S = m/m
    # tau_c = N/m^2; value from https://pubs.usgs.gov/sir/2008/5093/table7.html 
    #     =====> 0.0091 mm is avg
    L, rho_w, rho_s, g, S, tau_c, d50, d95 = [4.57, 1000, 2650, 
                                              9.81, 0.03, 0.0630,
                                              1.56e-6, 0.0275]
    #===========================DEFINE LAYER CONSTANTS===========================
    # h_s = depth of surfacing
    # f_sf, f_sc = fractions of fine/coarse material in ballast
    # h_b = depth of surfacing
    # f_bf, f_bc = fractions of fine/coarse material in ballast
    h_s, f_sf, f_sc = [0.23, 0.275, 0.725]
    h_b, f_bf, f_br = [2, 0.20, 0.80]

    #===========================DEFINE PUMPING/CRUSHING RATES===========================
    # k_cs = crushing constant for surfacing, m/truck pass
    # k_cb = crushing constant for ballast, m/truck pass
    # u_ps = pumping constant for surfacing, m/truck pass
    #   (2.14e-5m^3/4.57 m^2)  
    #   6 tires * 0.225 m width * 0.005 m length * 3.175e-3 m treads
    # u_pb = pumping constant for ballast, m/truck pass
    # e = fraction of coarse material, -
#     k_cs, k_cb, u_ps, u_pb, e = [1e-7, 1e-7, 
#                                  1e-7, 1e-7, 
#                                  0.725] #e needs to be variable... right?

    #===========================GROUP RAINFALL DATA===========================
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


    #===========================INITIALIZE ARRAYS===========================
    #=================len(storms_df)====================
    dS_f = np.zeros(len(storms_df))
    S_f_init = np.zeros(len(storms_df))
    S_f = np.zeros(len(storms_df))
    S_s = np.zeros(len(storms_df))
    S_sc = np.zeros(len(storms_df))
    S_sf = np.zeros(len(storms_df))
    S_b = np.zeros(len(storms_df))
    S_bc = np.zeros(len(storms_df))
    S_bf = np.zeros(len(storms_df))
    Hs_out = np.zeros(len(storms_df))

    r_storm = np.zeros(len(storms_df))
    q_storm = np.zeros(len(storms_df))
    q_f1 = np.zeros(len(storms_df))
    q_f2 = np.zeros(len(storms_df))
    q_as = np.zeros(len(storms_df))
    q_ab = np.zeros(len(storms_df))
    sed_added = np.zeros(len(storms_df))
    sed_cap = np.zeros(len(storms_df))
    ref_trans = np.zeros(len(storms_df))
    # storms_df['q_storm'] = storms_df.intensity*2.77778e-7*L 
    # q_storm = storms_df.q_storm.to_numpy()
    q_s_avg = np.zeros(len(storms_df))
    q_ref_avg = np.zeros(len(storms_df))

    n_tp = storms_df.truck_pass.to_numpy()
    t = storms_df.deltaT.to_numpy()

    #=================len(int_tip_df)====================
    q = np.zeros(len(int_tip_df))
    q_avg = np.zeros(len(int_tip_df))
    r_avg = np.zeros(len(int_tip_df))
    f_s = np.zeros(len(int_tip_df))
    n_f = np.zeros(len(int_tip_df))
    n_c = np.zeros(len(int_tip_df))
    n_t = np.zeros(len(int_tip_df))
    q_s = np.zeros(len(int_tip_df))
    q_ref = np.zeros(len(int_tip_df))
    water_depth = np.zeros(len(int_tip_df))
    tau = np.zeros(len(int_tip_df))
    tau_e = np.zeros(len(int_tip_df))

    rainfall = int_tip_df.intensity_mmhr.to_numpy()
    frac = int_tip_df.frac.to_numpy()
    stormNo = int_tip_df.stormNo.to_numpy()
    t_storm = int_tip_df.groupby('stormNo')['storm_dur'].mean().to_numpy()

    #===========================INITIALIZE DEPTHS===========================
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
        else:
            q_f1[j] = u_ps*(S_sf[j-1]/S_s[j-1])*n_tp[j]/(t[j]*3600)
            q_f2[j] = u_pb*(S_bf[j-1]/S_b[j-1])*n_tp[j]/(t[j]*3600)
            q_as[j] = k_cs*(S_sc[j-1]/S_s[j-1])*n_tp[j]/(t[j]*3600)
            q_ab[j] = k_cb*(S_bc[j-1]/S_b[j-1])*n_tp[j]/(t[j]*3600)
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

    #===========================BEGIN INTEGRATE OVER qs===========================
        for k, val in enumerate(stormNo):
            if k == 0:
                continue
            else:
                q[k] = rainfall[k]*2.77778e-7*L 
                q_avg[k] = q[k]*frac[k]
                r_avg[k] = rainfall[k]*frac[k]

                if q_avg[k] > 0:
                    n_f[k] = 0.0026*q_avg[k]**(-0.274)
                    n_c[k] = 0.08*q_avg[k]**(-0.153)
                else:
                    n_f[k] = n_f[k-1]
                    n_c[k] = n_c[k-1]

                if S_f_init[j] <= d95:
                    n_t[k] = n_c[k] + (S_f_init[j]/d95)*(n_f[k]-n_c[k])
                    f_s[k] = (n_f[k]/n_t[k])**(1.5)*(S_f_init[j]/d95)
                else: 
                    n_t[k] = n_f[k]
                    f_s[k] = (n_f[k]/n_t[k])**(1.5)

                #Calculate water depth assuming uniform overland flow
                water_depth[k] = ((n_t[k]*q_avg[k])/(S**(1/2)))**(3/5)

                tau[k] = rho_w*g*water_depth[k]*S
                tau_e[k] = tau[k]*f_s[k]

                #Calculate sediment transport rate
                if (tau_e[k]-tau_c) >= 0:
                    q_s[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*\
                             (tau_e[k]-tau_c)**(2.457))
                else:
                    q_s[k] = 0

                #Calculate reference transport 
                if (tau[k]-tau_c) >=0:
                    q_ref[k] = (((10**(-4.348))/(rho_s*((d50)**(0.811))))*\
                                   (tau[k]-tau_c)**(2.457))
                else:
                    q_ref[k] = 0


                if val == storm:
                    q_storm[j] += q_avg[k]
                    r_storm[j] += r_avg[k]
                    q_s_avg[j] += q_s[k]
                    q_ref_avg[j] += q_ref[k]

    #===========================END INTEGRATE OVER qs===========================

        sed_cap[j] = q_s_avg[j]*t_storm[j]*3600/L
        ref_trans[j] = q_ref_avg[j]*t_storm[j]*3600/L

        Hs_out[j] = np.minimum(sed_added[j]+S_f[j-1], sed_cap[j])
        dS_f[j] = sed_added[j] - Hs_out[j]
        S_f[j] = S_f[j-1] + dS_f[j]

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
    storms_df['q_storm'] = q_storm
    storms_df['r_storm'] = r_storm
    # storms_df['water_depth'] = water_depth
    # storms_df['tau'] = tau
    # storms_df['tau_e'] = tau_e
    # storms_df['n_t'] = n_t
    # storms_df['f_s'] = f_s
    # storms_df['qs'] = q_s

    int_tip_df['q'] = q
    int_tip_df['q_avg'] = q_avg

    sed_sum_m = storms_df.sed_added.sum()-(storms_df.Hs_out.sum()/1000)
    sed_sum_kg_m = sed_sum_m*rho_s*L
    f = ((storms_df.S_f[len(storms_df)-1]-storms_df.S_f[0])/1000)*rho_s*L

#     if round(f) == round(sed_sum_kg_m):
#         print('\nThe mass balance is fine.')
#     else:
#         print('\nThe mass balance is off.')

    total_out_kg = (storms_df.Hs_out.sum()/1000)*rho_s*L
    print("\nTotal amount of sediment transported:", round(total_out_kg), "kg/m")
    
    return(total_out_kg, storms_df)