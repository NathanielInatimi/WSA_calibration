# calibration functions

import datetime
import numpy as np
import pandas as pd
import xarray as xr
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
#from evaluation_functions import ensemble_evaluation_functions as eef
#import sunspots.sunspots as sunspots
import re  #for dealing with non-numeric characters in a string of unknown length
import os

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin

import scipy.interpolate
from scipy import integrate

from sunpy.coordinates.sun import carrington_rotation_time
from sunpy.coordinates.sun import carrington_rotation_number

# ensemble functions
import huxt_ensemble_functions as hef 
import time


def save_chi_arr_by_longitude_to_file(chi_set_list, era_key, sigma_longitude, lead_time):
    fname = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\WSA_calibration\\data\\rank_analysis\\rank_hist_{era_key}_{int(sigma_longitude)}_{lead_time}_lead.csv'
    np.savetxt(fname = fname, X = chi_set_list, delimiter = ',')
    return

def save_BS_by_longitude_to_file(BS_set_list, era_key, sigma_longitude, lead_time, threshold):
    fname = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\WSA_calibration\\data\\brier_scores\\BS_{era_key}_{int(sigma_longitude)}_{lead_time}_lead_{int(threshold)}.csv'
    np.savetxt(fname = fname, X = BS_set_list, delimiter = ',')
    return

def save_ROC_by_longitude_to_file(ROC_set_list, era_key, sigma_longitude, lead_time, probability_threshold):
    fname = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\WSA_calibration\\data\\roc_scores\\ROC_{era_key}_{int(sigma_longitude)}_{lead_time}_lead_{probability_threshold}.csv'
    np.savetxt(fname = fname, X = ROC_set_list, delimiter = ',')
    return

def save_REF_by_longitude_to_file(REF_set_list, era_key, sigma_longitude, lead_time, threshold):
    fname = f'BS_{era_key}_{int(sigma_longitude)}_{lead_time}_lead_{int(threshold)}.csv'
    file_path = os.path.abspath(os.path.join(os.pardir,'data','refinement_scores',fname))
    np.savetxt(fname = file_path, X = REF_set_list, delimiter = ',')
    return

def save_CAL_by_longitude_to_file(CAL_set_list, era_key, sigma_longitude, lead_time, threshold):
    fname = f'BS_{era_key}_{int(sigma_longitude)}_{lead_time}_lead_{int(threshold)}.csv'
    file_path = os.path.abspath(os.path.join(os.pardir,'data','calibration_scores',fname))
    np.savetxt(fname = file_path, X = CAL_set_list, delimiter = ',')
    return


def read_chi_arr_longitude_file(era_key, sigma_longitude, lead_time):
    fname = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\WSA_calibration\\data\\rank_analysis\\rank_hist_{era_key}_{sigma_longitude}_{lead_time}_lead.csv'
    return np.genfromtxt(fname = fname, delimiter=',')

def gen_ranked_ensemble(ensemble_members, observed_data): 

    """
    ranks an ensemble forecast (counts fraction of ensemble members overpredicting observed wind speed)

    Args:
        ensemble_members (list) : list of dataFrames of model runs, timeseries extracted at Earth
        observed_data (dataFrame) : observed near-Earth solar wind speed data
    
    Returns:
        summed_rank (array) : array with counts of ensemble members overprediscting windspeed per timestep
    """

    # Compare ensemble member output arrays to omni data 
    vsw_arr = ensemble_members
    ranked_forecast_boolean = np.array([vsw < observed_data for vsw in vsw_arr])
    summed_ranks = np.sum(ranked_forecast_boolean, axis = 0)

    return summed_ranks

def get_filenames_between_dates(start_date, end_date, sigma_latitude, ensemble_size, year_str):

    year_id = {'2023':'', '2020':'_2020'} # ID for finding files run for 2023/2020 WSA solutions (bit of a bodge should standardise)

    # get all ensemble netCDF file strings of specified parameters
    ensemble_directory_path = os.path.abspath(os.path.join(os.pardir,'data','ensembles'))
    ensemble_file_words = [f'ens_{int(sigma_latitude)}_{ensemble_size}{year_id[year_str]}'] # keywords to filter ensemble set directory
    ensemble_fname = hef.get_files_containing_words(ensemble_directory_path, ensemble_file_words)[0]

    all_files = os.listdir(os.path.abspath(os.path.join(os.pardir,'data','ensembles',ensemble_fname))) # list of all files inside ensemble directory

    dates = []

    # creating list of filenames of ensemble files for reading in ensembles over a specfied date range
    for filename in all_files:

        #unpack ensemble datenum into an indexable datetime obj
        pattern = r'_(\d{4})(\d{2})(\d{2})(\d{2})'
        match = re.search(pattern, filename)
        year, month, day, hour = match.groups()
        date_string = f'{year}-{month}-{day}--{hour}'
        dates.append(datetime.datetime(int(year), int(month), int(day), int(hour)))

    # index ensemble filenames by date
    df_ens_filenames = pd.DataFrame({'file_string' : all_files}, index = dates)
    df_ens_filenames = df_ens_filenames.sort_index()

    date_range = pd.date_range(start_date, end_date, freq='D') # daily frequency

    # Finding closest indices
    indexer = df_ens_filenames.index.get_indexer(date_range, method='nearest')

    # Retrieving the closest rows
    closest_files = df_ens_filenames.iloc[indexer]

    # Dropping duplicates to keep only unique rows -- unlikely for any duplicates but just in case :/ 
    unique_files = closest_files[~closest_files.index.duplicated(keep='first')]

    # list of ensemble netCDF filenames within date_range
    fname_list = unique_files['file_string'].to_list()

    return fname_list

def date_from_ensemble_folder_name(fname):
    
    """
    extracts date from wsa filename

    Args:
        fname (string) : name of ensemble file set
    Returns:
        date_string (string) : date string generated from info within file string format
        date_obj (datetime) : datetime object from file name
    """

    pattern = r'_(\d{4})(\d{2})(\d{2})(\d{2})'
    match = re.search(pattern, fname)
    year, month, day, hour = match.groups()
    date_string = f'{year}-{month}-{day}--{hour}'
    date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d--%H')

    return date_string, date_obj

def split_by_lead_time(params, sigma_longitude, max_lead_time, observed_data, filenames):

    """
    longitudinally perturbs and splits a set of forecasts by lead time into a dictionary

    Args:
        params (tuple) : contains information that was needed to generate ensemble (now used to read in file)
        sigma_longitude (float) : scale paramater which controls spread of longotudinal perturbation in degrees
        max_lead_time (int) : maximum lead time which forecasts will be split up to
        observed_data (dataFrame) : near-Earth wind speed obersvations (OMNI)
        filenames (list) : list of file

        ----------------------- PARAMS FORMAT ------------------------
        format : (filname, ensemble_size, sigma_latitude, forecast_window, r_min) 
        filename (string) : name of coronal model file name
        ensemble_size (int) : number of ensemble members
        sigma_latitude (float) : scale paramater which controls spread of intial conditions in degrees
        forecast_window (float) : duration of model run/forecast length in units of days
        r_min (float) : distance of inner boundary from Sun in units of solar radii 
        
        
    Returns:
        lead_time_dict (dictionary) : 
    """

    # Extract ensemble information (only ensemble size and sigma latitude needed)
    #filename = params[0]
    ensemble_size = params[1]
    sigma_latitude = params[2]
    #forecast_window = params[3]
    #r_min = params[4] # WSA at 21.5 rS

    # intialise and prepare dictionary to collate forecast lead time sets
    lead_time_dict = {}
    for i in range(max_lead_time):
        lead_time_dict.update({f'{i+1}_day_lead':[],
                            f'{i+1}_day_data': []})

    for fname in filenames:    

        # random seed generation
        date_string, date_obj = date_from_ensemble_folder_name(fname)
        random_seed = int(date_obj.strftime("%y%m%d%H%M"))

        # read in ensemble members
        ensemble_members = hef.read_ens_cdf(date_string=date_string, sigma_latitude=sigma_latitude, ensemble_size=ensemble_size, coronal_model='wsa')

        # longitudinal perturbation of all ensemble members
        lp_ens_members = hef.perturb_ensemble_longitudinally(ensemble_members=ensemble_members, sigma_longitude=sigma_longitude, 
                                                            ensemble_size=ensemble_size, random_seed=random_seed)

        # get data for relevant time chunk
        data_chunk = observed_data.loc[lp_ens_members[0].index[0]:lp_ens_members[0].index[-1]]
        data_chunk = data_chunk.dropna(subset = ['V']) # Remove rows with NaN values

        # Resampling ensemble members onto OMNI datastep
        resampled_ens = hef.resample_ensemble_members(ensemble_members=lp_ens_members, observed_data=data_chunk['V'])

        # start time from index
        init_time = resampled_ens[0].index[0]
        day_dt = pd.Timedelta(days=1)

        # loop through lead times and chunk up ensembles by lead time adding to lead time dictionary
        for i in range(max_lead_time):
            lead_time_dict[f'{i+1}_day_lead'].append(pd.concat([ens.loc[init_time+(day_dt*i):init_time+(day_dt*(i+1))] for ens in resampled_ens],
                                                                axis = 1, keys = np.arange(0,ensemble_size)))
            
            lead_time_dict[f'{i+1}_day_data'].append(data_chunk['V'].loc[init_time+(day_dt*i):init_time+(day_dt*(i+1))])

    return lead_time_dict

def ICMElist(filepath = None):
    """
    A script to read and process Ian Richardson's ICME list.

    Some pre-processing is required:
        Download the following webpage as a html file: 
            http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm
        Open in Excel, remove the year rows, delete last column (S) which is empty
        Cut out the data table only (delete header and footer)
        Save as a CSV.

    """
    
    # if filepath is None:
    #     datapath =  system._setup_dirs_()['datapath']
    #     filepath = os.path.join(datapath,
    #                             'icmetable.csv')
    
    
    icmes=pd.read_csv(filepath,header=None)
    #delete the first row
    icmes.drop(icmes.index[0], inplace=True)
    icmes.index = range(len(icmes))
    
    for rownum in range(0,len(icmes)):
        for colnum in range(0,3):
            #convert the three date stamps
            datestr=icmes[colnum][rownum]
            year=int(datestr[:4])
            month=int(datestr[5:7])
            day=int(datestr[8:10])
            hour=int(datestr[11:13])
            minute=int(datestr[13:15])
            #icmes.set_value(rownum,colnum,datetime(year,month, day,hour,minute,0))
            icmes.at[rownum,colnum] = datetime.datetime(year,month, day,hour,minute,0)
            
        #tidy up the plasma properties
        for paramno in range(10,17):
            dv=str(icmes[paramno][rownum])
            if dv == '...' or dv == 'dg' or dv == 'nan':
                #icmes.set_value(rownum,paramno,np.nan)
                icmes.at[rownum,paramno] = np.nan
            else:
                #remove any remaining non-numeric characters
                dv=re.sub('[^0-9]','', dv)
                #icmes.set_value(rownum,paramno,float(dv))
                icmes.at[rownum,paramno] = float(dv)
        
    
    #chage teh column headings
    icmes=icmes.rename(columns = {0:'Shock_time',
                                  1:'ICME_start',
                                  2:'ICME_end',
                                  10:'dV',
                                  11: 'V_mean',
                                  12:'V_max',
                                  13:'Bmag',
                                  14:'MCflag',
                                  15:'Dst',
                                  16:'V_transit'})
    return icmes

def compute_roc_score(roc_curve):
    """
    Computes integrates area under ROC curve using scipy quad returning integrated area as the ROC score

    Parameters:
    - roc_curve (list): list of tuples which each contain the hit rate and false alarm rate calculated at different thresholds

    Returns:
    - result (float): ROC Score calculated as the integrated area under ROC curve
    """

    # Unpack and prepare roc curve data
    y,x = zip(*roc_curve)

    x = np.array([xx for xx in x])
    y = np.array([yy for yy in y])

    nan_mask_x = ~np.isnan(x)
    nan_mask_y = ~np.isnan(y)

    nan_mask = np.logical_and(nan_mask_x, nan_mask_y)

    x = x[nan_mask]
    y = y[nan_mask]

    # Interpolate the curve
    interp_function = scipy.interpolate.interp1d(x, y, kind='linear')

    # Define the integration limits
    a = min(x)
    b = max(x)

    # Perform the integration
    result, error = integrate.quad(interp_function, a, b)

    return result

def parallel_evaluation(eval_params):

    # unpacking ensemble parameters
    sigma_longitude = eval_params[0]
    latitudes_to_test = eval_params[1]
    start_date = eval_params[2]
    end_date = eval_params[3]
    ensemble_size = eval_params[4]
    forecast_window = eval_params[5]
    r_min = eval_params[6]
    max_lead_time = eval_params[7]
    lead_time = eval_params[8]

    #evaluation data + parameters
    cme_removed_data = eval_params[9]
    event_threshold = eval_params[10]
    probability_threshold = eval_params[11]

    print(f'now testing sigma_longitude = {sigma_longitude}...')
    chi_across_latitude = []
    BS_across_latitude = []
    ROC_across_latitude = []
    t0 = time.time()
    
    for sigma_latitude in latitudes_to_test:
        
        t1 = time.time()

        # list of ensemble netCDF filenames within date_range
        fname_list = get_filenames_between_dates(start_date, end_date, sigma_latitude, ensemble_size)
        
        # params formatted as a tuple to feed to ensemble gen function
        params = ('dummy_fname', ensemble_size, sigma_latitude, forecast_window, r_min)

        lead_time_dict = split_by_lead_time(params=params, sigma_longitude=sigma_longitude, 
                                            max_lead_time=max_lead_time, observed_data=cme_removed_data, 
                                            filenames=fname_list)

        df_combined = pd.concat(lead_time_dict[f'{lead_time}_day_lead'])
        df_data = pd.concat(lead_time_dict[f'{lead_time}_day_data'])

        # calculate calibration/performance metrics
        ranked_ensemble = gen_ranked_ensemble(ensemble_members=df_combined.to_numpy().T, observed_data=df_data)
        chi_sq = hef.calculate_rank_chi_square(ensemble_size=ensemble_size, ranked_forecasts=ranked_ensemble)
        brier_score = hef.ensemble_brier_score(ensemble_members=df_combined.to_numpy().T, observed_data=df_data.to_numpy(), event_threshold=event_threshold, ensemble_size=ensemble_size)
        roc_curve = hef.generate_roc_curve_from_ensemble(ensemble_members=df_combined.to_numpy().T, observed_data=df_data.to_numpy(), threshold_range=(300,800), threshold_num=10, probability_threshold=probability_threshold)
        roc_score = compute_roc_score(roc_curve=roc_curve)

        chi_across_latitude.append(chi_sq)
        BS_across_latitude.append(brier_score)
        ROC_across_latitude.append(roc_score)

        t2=time.time()
        print(f'sigma_latitude = {sigma_latitude} evaluated in {t2-t1:.2f} seconds')

    save_chi_arr_by_longitude_to_file(chi_set_list=chi_across_latitude, era_key='2023', sigma_longitude=sigma_longitude, lead_time=lead_time)
    save_BS_by_longitude_to_file(BS_set_list=BS_across_latitude, era_key='2023', sigma_longitude=sigma_longitude, lead_time=lead_time, threshold=event_threshold)
    save_ROC_by_longitude_to_file(ROC_set_list=ROC_across_latitude, era_key='2023', sigma_longitude=sigma_longitude, lead_time=lead_time, probability_threshold=probability_threshold)

    t4 = time.time()
    print(f'sigma_longitude = {sigma_longitude} set evaluated over {t4-t0:.2f} seconds')

    return

def compute_brier_components(forecasts, observations, num_bins=10):

    # Create bins for the forecast probabilities
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Digitize the forecast probabilities into bins
    bin_indices = np.digitize(forecasts, bins) - 1

    # Initialise arrays to store the average forecast probability and observed frequency
    avg_forecast_probs = np.zeros(num_bins)
    observed_frequencies = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # Calculate average forecast probability and observed frequency for each bin
    for k in range(num_bins):
        in_bin = bin_indices == k
        bin_counts[k] = np.sum(in_bin)
        if bin_counts[k] > 0:
            avg_forecast_probs[k] = np.mean(forecasts[in_bin])
            observed_frequencies[k] = np.mean(observations[in_bin])
    
    # # climatlogical base rate --> Used for three-component decomposition IMPLEMENT LATER!!!!
    # base_rate = np.mean(observations)
    
    # Calibration (Reliability) component
    calibration = np.sum(bin_counts * (avg_forecast_probs - observed_frequencies)**2) / len(forecasts)
    
    # Refinement (Resolution + Uncertainty) component
    refinement = np.sum(bin_counts * (observed_frequencies * (1 - observed_frequencies))) / len(forecasts)
    
    return calibration, refinement
