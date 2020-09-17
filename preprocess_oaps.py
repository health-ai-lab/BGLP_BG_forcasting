#!/usr/bin/env python
# coding: utf-8
# ---------------------------------------------------------------
# This code is used to:
# - extract gz files to json files
# - save json files as csv files
# - combine data from multiple files and save it as a single file 
#   for each subject (create multi-modality data)

# Author: Hadia Hameed
# References:
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://github.com/kleinberg-lab/FLK-NN
# ---------------------------------------------------------------

#datastructure packages
import numpy as np
import pandas as pd
from numpy import array

#file system packages
import sys
import os
from os import path
import glob
import warnings
warnings.filterwarnings('ignore')
import gzip

#datetime packages
import datetime
import time
import dateutil.parser

#miscellaneous
import math
from operator import add
import json
import scipy as sp
from scipy import signal
import pickle 
from sklearn.preprocessing import MinMaxScaler

feature = 'entries' #[entries,devicestatus,profile,treatments]

#extracts gz files and saves them as json files in "extracted_files" folder
#Also converts json files for each subject and saves them as csv files
def gz_to_json_csv_files():
    for subj in subjs:
        subj_data_dir = '%s%s/direct-sharing-31/' %(data_directory,subj) #e.g. /data/PHI/PHI_OAPS/OpenAPS_data/n=88_OpenAPSDataAugust242018/15634563/direct-sharing-31/
        fn = '%s*.json.gz'%(subj_data_dir) 
        if path.exists(subj_data_dir) == True:
            print('-------------------Extracting files for subj '+subj+'-------------------')
            files = glob.glob(fn) #get all gz.json files in the directory
            for fn in files:
                if os.path.isfile(fn):
                    filename = fn.split('/')[-1] #get filename
                    filename = filename.split('.')[0] #remove extension
                    
                    try:
                        with gzip.open(fn, "rt",  encoding="utf-8") as f:
                            data = json.load(f)
                        extracted_folder_dir = '%s%s/direct-sharing-31/extracted_files/' %(data_directory,subj)
                        if not path.exists(extracted_folder_dir):
                            os.mkdir(extracted_folder_dir)
                        filename = extracted_folder_dir+filename
                        #saving json file
                        with open(filename+'.json', 'w') as fp: 
                            json.dump(data, fp)
                        print('Json data for subject '+ subj + ' saved to '+filename+'\n')
                        #saving csv file
                        df = pd.DataFrame(data)
                        df.to_csv(filename+'.csv', encoding='utf-8')
                        print("Saving csv files for subject: " + subj + "\n")
                    except Exception as e:
                        print("error: {0}".format(e))

#returns data values from different files as a dataframe for a given subject "subj" and "feature" 
#i.e. feature = ['entries','treatments','devicestatus']
#e.g. if feature is entries, there can be multiple entries files. This function returns a list of
# dataframes read from each file for a given feature
def get_subject_data(subj):
    data_frames = list() #list of dataframes
    subj_data_file = '%s%s/direct-sharing-31/extracted_files/' %(data_directory,subj)
    if feature == 'entries':
        attributes = ['dateString','sgv','device']
    elif feature == 'treatments':
        attributes = ['created_at','insulin','absolute','carbs']
    elif feature == 'devicestatus' and 'openaps' in df.columns:
        attributes = ['openaps']

    if path.exists(subj_data_file) == True:
        files = glob.glob(subj_data_file+feature+"*.csv")
        files.sort()
        for f in files:
            df = pd.read_csv(f)
            if all(elem in df.columns for elem in attributes):
                if not df.empty:
                    if feature == 'entries':
                        df.drop_duplicates(subset=attributes[0:2], keep='first', inplace=True)
                        data_frames.append(df[attributes])
                    if feature == 'devicestatus' and 'openaps' in df.columns:
                        data_frames.append(df[attributes])
                    if feature == 'treatments':
                        data_frames.append(df[attributes])
            else:
                continue

    return data_frames

# Reference: https://github.com/kleinberg-lab/FLK-NN
def fourier_impute(mis_mat,des_percent=100):   
    sig_missing = mis_mat
    if any((np.isnan(mis_mat))):
        miss_st = np.isnan(sig_missing).argmax()
    else:
        miss_st = nan
    while not math.isnan(miss_st):
        if not any(~np.isnan(sig_missing[miss_st:])):
            miss_fi = mis_mat.shape[0]
        else:
            miss_fi = miss_st + (~np.isnan(sig_missing[miss_st:])).argmax() #number of consecutive missing values after the miss_st index 
        sig_segment = sig_missing[0:miss_st]
        if len(sig_segment) > 0:
            Fsig_segment = np.fft.fft(sig_segment)
            descriptor_len = math.ceil(len(Fsig_segment)*des_percent/100)
            temp_sig = np.fft.ifft(Fsig_segment[0:descriptor_len], miss_fi).real
            sig_missing[miss_st:miss_fi] = temp_sig[miss_st:miss_fi]
        else:
            sig_missing[miss_st:miss_fi] = 0
        if(any(np.isnan(sig_missing[miss_fi:]))):
            miss_st = miss_fi + np.isnan(sig_missing[miss_fi:]).argmax()
        else:
            miss_st = nan
    mis_mat = sig_missing
    return mis_mat

def parse_dates(df):
    try:
        attribute = df.columns[0]
        df[attribute ] = pd.to_datetime(df[attribute ],utc=True)
    except:
        for row in df.itertuples():
            date = str(df.at[row.Index, attribute ])
            if date.split(' ')[-1][-1] == 'M': #looks like 06/06/2016 22:53:33 PM
                date = ' '.join(date.split(' ')[:-1])
                date = pd.to_datetime(date)
            elif date.split('-')[0][0] == '0': #looks like 0117-07-31T05:01:13-05:00 for subject 97872409
                date = list(date)
                date[0] = '2'
                date[1] = '0'
                date = "".join(date)
            elif date.split('-')[0][0] == '1': #looks like 117-07-31T05:01:13-05:00 for subject 97872409
                temp = date.split('-') #['117', '07', '31T06:01:13', '04:00']
                v2 = '201'+temp[0][-1] #'2017'
                temp[0] = v2 #['2017', '07', '31T06:01:13', '04:00']
                date = '-'.join(temp)
            date = pd.to_datetime(date,utc=True)
            df.at[row.Index, attribute] = date.replace(tzinfo=None)
    df[attribute] = df[attribute].dt.tz_localize(None)
    df[attribute] = df[attribute].astype('datetime64[s]')
    df.sort_values(by=attribute)

    return df

#subjects often have CGM and other data stored in multiple files
#This combines data from different files for a given subject and stores it as a single file
#This assumes .csv files exist in the extracted_files folder for each subject
def combine_data():
    if feature == 'entries':
        attributes = ['dateString','sgv','device']
    elif feature == 'treatments':
        attributes = ['created_at','insulin','absolute','carbs']
    elif feature == 'devicestatus' and 'openaps' in df.columns:
        attributes = ['openaps']

    for subj in subjs:
        data_frames = get_subject_data(subj)
        combined_data = pd.DataFrame()
        print("Getting data info for subject: " + subj + "\n")
        for df in data_frames:
            if feature == 'entries':
                df = parse_dates(df)
            combined_data = pd.concat([combined_data,df],axis=0)
        if not combined_data.empty:
            combined_data.dropna(subset=attributes[0:1],inplace=True) #drop rows for which dates are missing
            combined_data = combined_data.drop_duplicates(attributes,keep='first') #drop rows for which timestamp and sgv values are equal
            combined_data['dateString'] = pd.to_datetime(combined_data[attributes[0]],utc=True,errors='coerce')
            combined_data.dropna(subset=attributes[0:1],inplace=True) #drop rows for which dates are missing after conversion
            combined_data['dateString'] = combined_data['dateString'].dt.tz_localize(None) #the timezone for treatments is different from that of entries
            combined_data['dateString'] = combined_data['dateString'].astype('datetime64[s]')
            
            subj_combined_data = combined_data.sort_values(by=attributes[0]) #sort by dates
            extracted_folder_dir = '%s%s/direct-sharing-31/extracted_files/complete_data_%s_%s.csv' %(data_directory,subj,feature,subj)
            subj_combined_data.to_csv(extracted_folder_dir)

#******************************* MAIN Functions for data subsetting ***************************************

# Helper Function: divides data into subsets based on time difference to ensure that there are not any long gaps
# within each subset. The function create_time_series() calls the following function
def process_multi_modality_data(df,subj,all_subjects_data,dataset):

    #replace all missing basal rates with the last recorded basal rate
    df['basal'] = df['basal'].fillna(method='ffill')
    
    #removing leadning NANs
    df = df[np.where(~df['glucose_level'].isnull())[0][0]:] 
    #Remove rows for which cgm value is missing for more than 5 rows (roughly 25 minutes)
    mask = df.glucose_level.notna()
    a = mask.ne(mask.shift()).cumsum()
    df = df[(a.groupby(a).transform('size') < 5) | mask]
    
    df.reset_index(inplace=True)
    df = df[['dates','glucose_level','basal','meal','bolus']]
    

    df["Time_diff"] = df['dates'].diff() #time difference between consecutive time stamps
    gaps = df[df["Time_diff"] > '00:30:00'] #indices where the time gaps are greater than 30 minutes 
    indices = gaps.index.to_series().values
    start_index = 0
    original_df = df.copy()

    #e.g. there are long gaps at indices [200,400]
    # in the first iteration get subset df[0:200], in the second iteration get df[200:400],
    # in the final iteration get df[400:]
    key = 0

    for i in range(len(indices)+1):
        if i < len(indices):
            subset_df = original_df.iloc[start_index:indices[i]]
        else:
            subset_df = original_df.iloc[start_index:]

        if len(np.where(~subset_df['glucose_level'].isnull())[0]) > 0:
            subset_df = subset_df[np.where(~subset_df['glucose_level'].isnull())[0][0]:]
    
        if subset_df.shape[0] < 2*history_window:
            if i < len(indices):
                start_index = indices[i] #move on to the next index
            continue


        #Interpolating CGM values using linear interpolation
        mask = subset_df.copy()
        grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
        grp['ones'] = 1
        mask['glucose_level'] = (grp.groupby('glucose_level')['ones'].transform('count') <= 5) | subset_df['glucose_level'].notnull()

        subset_df['meal'] = subset_df['meal'].fillna(0) #fill missing meals with 0 carb
        subset_df['bolus'] = subset_df['bolus'].fillna(0) #fill missing boluses with 0

        
        #imputing glucose values
        
        subset_df['glucose_level'] = subset_df['glucose_level'].astype('float32')
        if dataset == 'train':
            if sys.argv[-1] == '0' or sys.argv[-1] == '1':
                subset_df['glucose_level'] = subset_df['glucose_level'].interpolate()[mask['glucose_level']] #imputation technique 1: linear interpolation if training set
            else:
                subset_df = subset_df[subset_df['glucose_level'].notna()] #imputation technique 2: linear interpolation if training set
        elif dataset == 'test':
            if sys.argv[-1] == '2' or sys.argv[-1] == '3':
                subset_df = subset_df[subset_df['glucose_level'].notna()] #imputation technique 2: linear interpolation if training set
            else:
                subset_df['glucose_level'] = subset_df['glucose_level'].ffill()[mask['glucose_level']] #forward filling if test set (extrapolation)

        subset_df['glucose_level'] = subset_df['glucose_level'].astype('int32')
        
        #median filtering
        if filter_data:
            subset_df['glucose_level'] = sp.signal.medfilt(subset_df['glucose_level'].values,5) #median filtering 
        
        if i < len(indices):
            start_index = indices[i] #move on to the next index
        subset_df.dropna(subset=['basal'], inplace=True) #if basal values are missing before the first basal value is recorded
        subset_df.reset_index(inplace=True)

        if subset_df.shape[0] >= 2*history_window: #the minimum number of samples should be >= 2*history_window:

            subset_df["glucose_diff"] = subset_df['glucose_level'].diff() #time difference between consecutive time stamps
            subset_df['glucose_diff'] = subset_df['glucose_diff'].fillna(0) #replacing the NaT in the first row 
            all_subjects_data[subj][key] = subset_df
 
            key = key + 1

    return all_subjects_data

# create continuous time-series subsets. (n-1) years are stored as training data for subjects and nth year is stored as 
# test data for each subject
def create_time_series():
    subjs.sort()
    all_subjects_train_data = {}
    all_subjects_test_data = {}

    for subj in subjs:
        insulin_path = '%s%s/direct-sharing-31/extracted_files/complete_data_treatments_%s.csv' %(data_directory,subj,subj)
        cgm_path = '%s%s/direct-sharing-31/extracted_files/complete_data_%s.csv' %(data_directory,subj,subj)
        
        #cgm data or insulin data not present for subject, just continue
        if path.exists(insulin_path) == False or path.exists(cgm_path) == False: 
            continue

        # absolute = basal rate , insulin = bolus rate
        insulin_data =  pd.read_csv(insulin_path, usecols = ['dateString', 'absolute', 'insulin' , 'carbs'],parse_dates = True)
        cgm_data =  pd.read_csv(cgm_path, usecols = ['dateString', 'sgv'], parse_dates = True)

        if insulin_data.empty or cgm_data.empty:
            continue

        print('-------------Imputing data for Subject: '+subj+'-------------')
        insulin_data.set_index('dateString', inplace=True)
        cgm_data.set_index('dateString', inplace=True)

        original_df = insulin_data.join(cgm_data, how='outer') #concatenate two time series based on timestamps

        original_df.reset_index(inplace=True)
        original_df.rename(columns={'dateString':'dates', 'insulin':'bolus', 'absolute':'basal', 'sgv':'glucose_level', 'carbs':'meal'},inplace = True)
        original_df['dates']= pd.to_datetime(original_df['dates']) #dates are initially stored as strings. Convert them to dateTime
        original_df.sort_values(by='dates',inplace=True) #sort in the ascending order of date and time to maintain natural temporal ordering


        original_df['glucose_level'].loc[original_df['glucose_level'] <= threshold] = np.nan #Thresholding
        #group by year
        original_df['year'] = [d.year for d in original_df['dates']]  #group by year 
        grouped = original_df.groupby(original_df['year'])
        n = grouped.ngroups #number of years present

        if n > 1: #if there are more than one year of data available for the subject
            train_data = pd.DataFrame()
            test_data = pd.DataFrame()
            year_counter = 1 #keep a count of the years present
            all_subjects_train_data[subj] = {}
            all_subjects_test_data[subj] = {}
            for name, group in grouped:
                if year_counter < n: #all n-1 years go to train data

                    if group.shape[0] >= 2*history_window:
                        train_data = pd.concat([train_data,group],axis=0)
                        year_counter = year_counter + 1
                    else:
                        print("Data for year "+ str(name) +" too small in size")
                        year_counter = year_counter + 1
                        if train_data.empty and year_counter == n: #if n years were present and n-1 were too small to add to train data, don't add it to test set either
                            year_counter = year_counter + 1
                            print('Removing subj '+subj)
                            del all_subjects_train_data[subj]
                            del all_subjects_test_data[subj]
                elif year_counter == n: #last year goes to test data
                    if group.shape[0] >= 2*history_window:
                        test_data = group
                    else:
                        print("Data for year "+ str(name) +" too small in size") #if the last year's data is too small to put in the test set, remove the train set for that subject
                        print('Removing subj '+subj)
                        del all_subjects_train_data[subj]
                        del all_subjects_test_data[subj]
        else:

            print('Less than a year of data present')
            continue

        if subj in all_subjects_train_data.keys():
            all_subjects_train_data = process_multi_modality_data(train_data,subj,all_subjects_train_data,'train')
            all_subjects_test_data = process_multi_modality_data(test_data,subj,all_subjects_test_data,'test')

            if not all_subjects_train_data[subj] or not all_subjects_test_data[subj]: #if the combined data was empty after processing
                print('Not enough data. Removing subj '+subj)
                del all_subjects_train_data[subj]
                del all_subjects_test_data[subj]



    if not path.exists(data_directory):
        os.mkdir(data_directory)
    with open(data_directory + 'multivariate_continuous_train_subsets.pickle', 'wb') as f:
        pickle.dump(all_subjects_train_data, f, pickle.HIGHEST_PROTOCOL)
    with open(data_directory + 'multivariate_continuous_test_subsets.pickle', 'wb') as f:
        pickle.dump(all_subjects_test_data, f, pickle.HIGHEST_PROTOCOL)

#******************************* MAIN Functions for windowing data ***************************************
# convert series to supervised learning
# Reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# if only CGM values are required (univariate series), use series = 'uni', otherwise series = 'multi'
# HELPER windowing FUNCTION
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    features = ['date','CGM','basal','meal','bolus','glucose_diff']
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in features]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[1].shift(-i))
        if i == 0:
            names += [('CGM(t)')]
        else:
            names += [('CGM(t+%d)' % i)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# HELPER windowing FUNCTION
def window_data(filePath):
    with open(filePath, 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
    
    subjs = unpickled_data.keys()
    all_windowed_data = {}

    for subj in subjs:   
        print('-----------------Windowing data for Subject: '+subj+'-------------------')    
        all_windowed_data[subj] = {}
        keys = list(unpickled_data[subj].keys())
        reframed = pd.DataFrame()
        for key in keys:
            df = unpickled_data[subj][key]
            df = df[['dates', 'glucose_level','basal','meal', 'bolus','glucose_diff']]
            df.sort_values(by='dates',inplace=True) #sort in the ascending order of date and time to maintain natural temporal ordering
            
            #df.set_index('dates',inplace=True)
            
            df['meal'] = df['meal'].fillna(0)
            df['bolus'] = df['bolus'].fillna(0)

            df['basal'] = df['basal'].fillna(method='ffill')

            #meal_noise = np.random.normal(0,1,(df['meal']==0).sum())
            #df.loc[df['meal'] == 0, 'meal'] = meal_noise
            #bolus_noise = np.random.normal(0,1,(df['bolus']==0).sum())
            #df.loc[df['bolus'] == 0, 'bolus'] = bolus_noise
            if normalize_data:
                glucose_meal_values = df[['basal','meal', 'bolus']].values
                min_val = df['glucose_level'].values.min()
                max_val = df['glucose_level'].values.max()

                if min_val == max_val:
                    min_val = max_val // 2
                scaler = MinMaxScaler(feature_range=(min_val,max_val))
                glucose_meal_values = scaler.fit_transform(glucose_meal_values)
                df[['basal','meal', 'bolus']] = glucose_meal_values

            values = df.values

            
            values = df.values
            reframed = pd.concat([reframed, series_to_supervised(values, history_window, prediction_window)])

        all_windowed_data[subj] = reframed
    return all_windowed_data

# main windowing FUNCTION
def get_windowed_data():
    train_data = window_data(data_directory + 'multivariate_continuous_train_subsets.pickle')
    test_data = window_data(data_directory + 'multivariate_continuous_test_subsets.pickle')
    
    PH = str(prediction_window * 5) #prediction horizon
    if normalize_data:
        substring = 'normalized_'+PH+'min.pickle'
    else:
        substring = PH+'min.pickle'
    with open(data_directory + 'windowed_train_' + substring, 'wb') as f:
        # Pickle the 'all_subjects_data' dictionary using the highest protocol available.
        pickle.dump(train_data , f, pickle.HIGHEST_PROTOCOL)
    with open(data_directory + 'windowed_test_' + substring, 'wb') as f:
        # Pickle the 'all_subjects_data' dictionary using the highest protocol available.
        pickle.dump(test_data , f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    root_directory = sys.argv[1]
    data_directory = sys.argv[2]
    filter_data = sys.argv[3]
    if filter_data == 'True':
        filter_data = True
    else:
        filter_data = False

    normalize_data = sys.argv[4] # scale up all features except glucose
    if normalize_data == 'True':
        normalize_data = True
    else:
        normalize_data = False

    threshold = int(sys.argv[5]) #remove CGM readings below 15 mg/dL
    history_window = int(sys.argv[6]) #no. of past values to use to make estimations of future values
    PH = sys.argv[7] #prediction horizon
    prediction_window = int(sys.argv[7]) #no. of future values to predict (6 denotes a prediction horizon of 5 * 6 = 30 min)
        
    if prediction_window == 30 or prediction_window == 60:
        prediction_window = prediction_window//5

    subjs = [subject for subject in os.listdir(data_directory) if "." not in subject]
    subjs.sort()
    gz_to_json_csv_files()
    combine_data()
    create_time_series()
    get_windowed_data()