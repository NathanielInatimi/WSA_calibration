
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

    latitudes_to_run = [0,5]

    t0 = time.time()

    for sigma_latitude in latitudes_to_run:
    
        # get all WSA files with specified keywords
        directory_path = H._setup_dirs_()['boundary_conditions']
        wsa_file_words = ['gongz'] # keywords to filter for in coronal model file directory
        wsa_fnames = hef.get_files_containing_words(directory_path, wsa_file_words)

        dates = []
        filenames = []

        # creating list of filenames of WSA solutions for generating/reading in ensembles
        for filename in wsa_fnames:

            # Define regular expression patterns to extract the date from file string
            pattern = r'21.5rs_(\d{4})(\d{2})(\d{2})(\d{2})'
            pattern2 = r'%2F(\d{4})%2F(\d{1,2})%2F(\d{1,2})%2F'
            pattern3 = r'gong_(\d{4})(\d{2})(\d{2})(\d{2})'
            pattern4 = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})R000_gongz' 

            # Match patterns for different WSA file string formats
            match = re.search(pattern, filename)
            match2 = re.search(pattern2, filename)
            match3 = re.search(pattern3, filename)
            match4 = re.search(pattern4, filename)
            
            
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
            elif match3:
                year, month, day, hour = match3.groups()
                date_string = f'{year}-{month}-{day}--{hour}'
                dates.append(datetime.datetime(int(year), int(month), int(day), int(hour)))
                filenames.append(filename)
            elif match4:
                year, month, day, hour, _ = match4.groups()
                date_string = f'{year}-{month}-{day}--{hour}'
                dates.append(datetime.datetime(int(year), int(month), int(day), int(hour)))
                filenames.append(filename)
            else:
                print(f"No date found in the string: {filename}")

        # index filenames by date
        df_filenames = pd.DataFrame({'file_string' : filenames}, index = dates)
        df_filenames = df_filenames.sort_index()

        # specify date range of WAS solutions to generate ensembles for
        start_date = datetime.datetime(2020,1,1)
        end_date = datetime.datetime(2021,1,1)

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

        # Define ensemble params
        ensemble_size = 10
        forecast_window = 10 * u.day
        r_min = 21.5*u.solRad
        year = 2020

        #create sets of input params for parallel processing
        input_params = [(fname, ensemble_size, sigma_latitude, forecast_window, r_min, year) for fname in fname_list]

        print(f'sigma latitude = {sigma_latitude}: parameters initialised')

        t1 = time.time()

        # initialise parallel processing for ensemble generation
        #multiprocessing.set_start_method('spawn')

        with multiprocessing.Pool(processes=4) as pool:
            pool.map(hef.generate_ensemble_forecast, input_params)

        t2 = time.time()

        print(f'{len(fname_list)} size {ensemble_size} ensembles took {t2-t1:.2f} seconds to generate ({(t2-t1)/60:.2f} mins)')
    
    t4 = time.time()
    print(f'Ensembles for {len(latitudes_to_run)} latitude scale parameter generated which took {(t4-t0)/60:2f} minutes')