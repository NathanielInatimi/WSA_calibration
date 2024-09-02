## ensemble_calibration_analyisis
import multiprocessing
import huxt as H
import huxt_ensemble_functions as hef
import calibration_functions as cf
import numpy as np
import pandas as pd
import astropy.units as u
import datetime
import re
import time

# omni data directory
omni_data_dir = 'C:\\Users\\ct832900\\Desktop\\Research_Code\\WSA_calibration\\data\\OMNI\\Processed_omni\\'

# load the data into dataframe and index by datetime
omni_data = pd.read_hdf(omni_data_dir + 'omni_1hour.h5')
omni_data = omni_data.set_index('datetime')
omni_data = omni_data.dropna(subset = ['V']) # Remove rows with NaN values

## Create verificaton dataset with ICMEs removed from the timeseries
df_ICME = cf.ICMElist("C:\\Users\\ct832900\\Desktop\\Research_Code\\WSA_Calibration\\data\\icmetable.csv")
df_ICME = df_ICME.set_index('Shock_time')

# Create a list of ICME (start,end) times 
CME_flags = [*zip(df_ICME.loc['2023']['ICME_start'], df_ICME.loc['2023']['ICME_end'])]

# Loop through ICME timing list and remove observations within CME start/end crossing period.
df_temp = omni_data.copy()
for start_time, end_time in CME_flags:

    # Replace values with NaN for rows within the specified time period
    df_temp.loc[(df_temp.index >= start_time) & (df_temp.index <= end_time)] = np.nan

    # Remove from time_series 
    #df_filtered = df_temp[~((df_temp.index >= start_time) & (df_temp.index <= end_time))]
    #df_temp = df_filtered.copy()

df_filtered = df_temp.copy()  
# Remove rows with NaN values from CME filtered verification dataset
cme_removed_data = df_filtered.dropna(subset = ['V']) 


# perturbation scale parameters to evaluate
# latitudes_to_test = np.linspace(0,40,41)
# longitudes_to_test = np.linspace(0,40,41)

latitudes_to_test = [1]
longitudes_to_test = [5]

# specify date range of ensemble to load in
start_date = datetime.datetime(2023,1,1)
end_date = datetime.datetime(2024,1,1)

# ensemble parameters
ensemble_size = 100
forecast_window = 10 * u.day
r_min = 21.5*u.solRad
max_lead_time = 1 # day(s)
lead_time = 1 # day(s)

# evaluation params
event_threshold = 385 #km/s
probability_threshold = 0.5 # for ROC curve

for sigma_longitude in longitudes_to_test:
    print(f'now testing sigma_longitude = {sigma_longitude}...')
    chi_across_latitude = []
    BS_across_latitude = []
    ROC_across_latitude = []
    t0 = time.time()
    
    for sigma_latitude in latitudes_to_test:
        
        t1 = time.time()

        # list of ensemble netCDF filenames within date_range
        fname_list = cf.get_filenames_between_dates(start_date, end_date, sigma_latitude, ensemble_size)
        
        # params formatted as a tuple to feed to ensemble gen function
        params = ('dummy_fname', ensemble_size, sigma_latitude, forecast_window, r_min)

        lead_time_dict = cf.split_by_lead_time(params=params, sigma_longitude=sigma_longitude, 
                                            max_lead_time=max_lead_time, observed_data=cme_removed_data, 
                                            filenames=fname_list)

        df_combined = pd.concat(lead_time_dict[f'{lead_time}_day_lead'])
        df_data = pd.concat(lead_time_dict[f'{lead_time}_day_data'])

        # calculate calibration/performance metrics
        ranked_ensemble = cf.gen_ranked_ensemble(ensemble_members=df_combined.to_numpy().T, observed_data=df_data)
        chi_sq = hef.calculate_rank_chi_square(ensemble_size=ensemble_size, ranked_forecasts=ranked_ensemble)
        brier_score = hef.ensemble_brier_score(ensemble_members=df_combined.to_numpy().T, observed_data=df_data.to_numpy(), event_threshold=event_threshold, ensemble_size=ensemble_size)
        roc_curve = hef.generate_roc_curve_from_ensemble(ensemble_members=df_combined.to_numpy().T, observed_data=df_data.to_numpy(), threshold_range=(300,800), threshold_num=10, probability_threshold=probability_threshold)
        roc_score = cf.compute_roc_score(roc_curve=roc_curve)

        chi_across_latitude.append(chi_sq)
        BS_across_latitude.append(brier_score)
        ROC_across_latitude.append(roc_score)

        t2=time.time()
        print(f'sigma_latitude = {sigma_latitude} evaluated in {t2-t1:.2f} seconds')

    cf.save_chi_arr_by_longitude_to_file(chi_set_list=chi_across_latitude, era_key='2023', sigma_longitude=sigma_longitude, lead_time=lead_time)
    cf.save_BS_by_longitude_to_file(BS_set_list=BS_across_latitude, era_key='2023', sigma_longitude=sigma_longitude, lead_time=lead_time, threshold=event_threshold)
    cf.save_ROC_by_longitude_to_file(ROC_set_list=ROC_across_latitude, era_key='2023', sigma_longitude=sigma_longitude, lead_time=lead_time, probability_threshold=probability_threshold)

    t4 = time.time()
    print(f'sigma_longitude = {sigma_longitude} set evaluated over {t4-t0:.2f} seconds')
