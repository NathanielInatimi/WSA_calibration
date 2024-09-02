# parallel ensemble evalutation

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


if __name__ == "__main__":

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

    df_filtered = df_temp.copy()  
    # Remove rows with NaN values from CME filtered verification dataset
    cme_removed_data = df_filtered.dropna(subset = ['V']) 

    # perturbation scale parameters to evaluate
    # latitudes_to_test = np.linspace(0,40,41)
    # longitudes_to_test = np.linspace(0,40,41)

    latitudes_to_test = [0,5]
    longitudes_to_test = [0,5]

    # specify date range of ensemble to load in
    start_date = datetime.datetime(2023,1,1)
    end_date = datetime.datetime(2024,1,1)

    # ensemble parameters
    ensemble_size = 100
    forecast_window = 10 * u.day
    r_min = 21.5*u.solRad
    max_lead_time = 1 # day(s)
    lead_time = 1 # day(s)

    event_threshold = 385 # km/s
    probability_threshold = 0.5 # for ROC curve

    input_params = [(sigma_longitude, latitudes_to_test, 
                     start_date, end_date, ensemble_size, forecast_window,
                     r_min, max_lead_time, lead_time, 
                     cme_removed_data, event_threshold, probability_threshold) 
                     for sigma_longitude in longitudes_to_test]
    
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(cf.parallel_evaluation, input_params)


