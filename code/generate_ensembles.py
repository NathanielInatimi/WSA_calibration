
import multiprocessing
import huxt as H
import huxt_ensemble_functions as hef
import numpy as np
import pandas as pd
import astropy.units as u
import datetime
import re
import time

#def main():

if __name__ == "__main__":
    
    # get all WSA files with specified keywords
    directory_path = H._setup_dirs_()['boundary_conditions']
    wsa_file_words = ['wsa_vel'] # keywords to filter for in coronal model file directory
    wsa_fnames = hef.get_files_containing_words(directory_path, wsa_file_words)

    dates = []
    filenames = []

    # creating list of filenames of WSA solutions for generating/reading in ensembles
    for filename in wsa_fnames:

        # Define regular expression patterns to extract the date from file string
        pattern = r'21.5rs_(\d{4})(\d{2})(\d{2})(\d{2})'
        pattern2 = r'%2F(\d{4})%2F(\d{1,2})%2F(\d{1,2})%2F'
        match = re.search(pattern, filename)
        match2 = re.search(pattern2, filename)
        
        if match:
            year, month, day, hour = match.groups()
            date_string = f'{year}-{month}-{day}--{hour}'
            dates.append(datetime.datetime(int(year), int(month), int(day), int(hour)))
            filenames.append(filename)
        elif match2:
            year, month, day = match2.groups()
            date_string = f'{year}-{month}-{day}'
            dates.append(datetime.datetime(int(year), int(month), int(day), int(0)))
            filenames.append(filename)
        else:
            print(f"No date found in the string: {filename}")

    # index filenames by date
    df_filenames = pd.DataFrame({'file_string' : filenames}, index = dates)
    df_filenames = df_filenames.sort_index()

    # specify date range of WAS solutions to generate ensembles for
    start_date = datetime.datetime(2019,10,10)
    end_date = datetime.datetime(2019,10,30)

    # want only 1 solution per day/as close to daily as possible
    date_range = pd.date_range(start_date, end_date, freq='D') 

    # Finding closest indices
    indexer = df_filenames.index.get_indexer(date_range, method='nearest')

    # Retrieving the closest rows
    closest_files = df_filenames.iloc[indexer]

    # Dropping duplicates to keep only unique rows
    unique_files = closest_files[~closest_files.index.duplicated(keep='first')]

    # list of WSA filenames within date_range
    fname_list = unique_files['file_string'].to_list()

    #ensemble params
    ensemble_size = 100
    sigma_latitude = 10 # degrees
    forecast_window = 10 * u.day
    r_min = 21.5*u.solRad

    #create sets of input params for parallel processing
    input_params = [(fname, ensemble_size, sigma_latitude, forecast_window, r_min) for fname in fname_list]

    print('parameters initialised')

    t1 = time.time()

    # initialise parallel processing for ensemble generation
    #multiprocessing.set_start_method('spawn')

    with multiprocessing.Pool(processes=4) as pool:
        pool.map(hef.generate_ensemble_forecast, input_params)

    t2 = time.time()

    print(f'{len(fname_list)} size {ensemble_size} ensembles took {t2-t1:.2f} seconds to generate ({(t2-t1)/60:.2f} mins)')
    