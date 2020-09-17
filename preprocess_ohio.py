#!/usr/bin/env python
# coding: utf-8
# ---------------------------------------------------------------
# This code converts performs the following tasks for OHIO data set
# - converts xml file to csv files
# - imputes CGM, basal rate and meal data
# - windows data

# Usage: python preprocess.py xml_to_csv
#     OR python preprocess.py impute_data
#     OR python preprocess.py get_windowed_data
#     OR python preprocess.py all (runs all the functions sequentially)

# Author = Hadia Hameed
# Refrences = https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# ---------------------------------------------------------------

#datastructure packages
import numpy as np
import pandas as pd
from numpy import array
import xml.etree.ElementTree as ET


#file system packages
import sys
import os
from os import path
import glob
import warnings
warnings.filterwarnings('ignore')
import gzip

#plotting packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.0)

#datetime packages
import datetime
import time
import dateutil.parser
import pytz

#miscellaneous
import math
from operator import add
import json
import random
import pickle
import scipy as sp
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

all_subjects_data = {}
windowed_data = {}
features = ['glucose_level','basal','meal','bolus','temp_basal']
subjs_paths = glob.glob(sys.argv[2] + ohio_data + '/*.xml') #e.g. ../../../data/PHI/PHI_OHIO/data/OhioT1DM-training
subjs = []

for subj_path in subjs_paths:
    subj = subj_path.split('/')[-1] #e.g. Filename = 540-ws-training.xml
    subj = subj.split('-')[0] #e.g. Subject ID = 540
    subjs.append(subj)

#Converting XML to CSV. Combining data for different variables
# parse an xml file by name
def xml_to_csv(subj):
    if ohio_data == 'OhioT1DM-training':
        doc = ET.parse(data_directory+ohio_data+'/'+subj+'-ws-training.xml') #e.g. '../../../data/PHI/PHI_OHIO/data/OhioT1DM-training/540-ws-training.xml'
    elif ohio_data == 'OhioT1DM-testing':
        doc = ET.parse(data_directory+ohio_data+'/'+subj+'-ws-testing.xml') #e.g. '../../../data/PHI/PHI_OHIO/data/OhioT1DM-training/540-ws-training.xml'
    print('-------------XML to CSV for Subject: '+subj+'-------------')
    root = doc.getroot() #<patient id="540" weight="99" insulin_type="Humalog"> gives root key i.e. "patient"
    all_variables = list()

    #List of all 19 variables
    #glucose_level, finger_stick, basal, temp_basal, bolus, meal, sleep, work, stressors, hypo_event
    #illness, exercise, basis_heart_rate, basis_gsr, basis_skin_temperature, basis_air_temperature
    #basis_steps, basis_sleep, acceleration
    for child in root:
        all_variables.append(child.tag) 
        
    df_list = list()
    
    for feature in features:
        dates = list()
        value = list()
        bolus_type = list()
        for row in root[all_variables.index(feature)]: #e.g each entry in a given tag like 'glucose_level' is a dictionary with <date> and <value>
            signal = row.attrib #e.g. {'ts': '07-12-2021 16:29:00', 'value': '101'}
            signal_keys = list(signal.keys()) #keys: [ <ts>, <value> ]

            if feature == 'temp_basal' or (feature == 'bolus' and 'square dual' in signal_keys):
   
                start_date = datetime.datetime.strptime(signal['ts_begin'],'%d-%m-%Y %H:%M:%S')
                end_date = datetime.datetime.strptime(signal['ts_end'],'%d-%m-%Y %H:%M:%S')
                elapsed_time = (end_date - start_date).total_seconds()/60.0 #number of minutes
                if 'dose' in signal_keys:
                    raw_value = float(signal[('dose')])
                else:
                    raw_value = float(signal[('value')])
  
                if elapsed_time > 0:
                    rate = 5*raw_value/elapsed_time #value absorbed every 5 minutes
                else:
                    rate = raw_value
                interpolated_dates = pd.date_range(start=start_date, end=end_date, freq='5T') #5-minute intervals
                dates.extend(list(interpolated_dates.values))
                for i in range(len(list(interpolated_dates.values))):
                    if i == 0:
                        if feature == 'bolus':
                            bolus_type.append(signal['type']+' start')
                        value.append(0.00) #no amount absorbed at the first time instance
                    elif i < len(list(interpolated_dates.values)) - 1:
                        if feature == 'bolus':
                            bolus_type.append(signal['type'])
                        value.append(rate)
                    else:
                        if feature == 'bolus':
                            bolus_type.append(signal['type']+' end')
                        value.append(rate)    
            else:
                if feature == 'bolus':
                    bolus_type.append(signal['type'])
                if 'ts_begin' in signal_keys:
                    dates.append(datetime.datetime.strptime(signal['ts_begin'],'%d-%m-%Y %H:%M:%S'))
                else:
                    dates.append(datetime.datetime.strptime(signal['ts'],'%d-%m-%Y %H:%M:%S'))
                if 'dose' in signal_keys:
                    value.append(signal['dose'])
                elif 'carbs' in signal_keys:
                    value.append(signal['carbs'])
                else:
       
                    value.append(signal['value'])
            
        if feature == 'bolus':
            df = pd.DataFrame(list(zip(dates,bolus_type,value)), columns = ['dates','bolus_type',feature])
        else:
            df = pd.DataFrame(list(zip(dates,value)), columns = ['dates',feature])
        df.drop_duplicates(subset=['dates'], keep='first', inplace=True)
        #df['dates'] = pd.to_datetime(df['dates'],format='%d-%m-%Y %H:%M:%S')
        df.set_index('dates',inplace=True)
        df_list.append(df) #list of dataframes for each variable i.e. glucose_value, basal etc.
    
    #union of dates in each dataframe
    newindex = df_list[len(df_list)-1].index
    for ii in range(len(df_list)-2,-1,-1):
        newindex = df_list[ii].index.union(newindex)

    #newindex = df_list[0].index.union(df_list[1].index)

    #concatenate dataframes column-wise for all the variables
    for ii in range(len(df_list)):
        df_list[ii] = df_list[ii].reindex(newindex)

    df = pd.concat(df_list[:], join='outer',axis=1,sort='False')

    df.sort_values(by='dates',inplace=True)
    return df

#parses dates and groups time series by year
def preprocess_data(df):
    dates = list()
    times = list()
    df['dates']= pd.to_datetime(df['dates'], format='%d-%m-%Y %H:%M:%S') #dates are initially stored as strings. Convert them to dateTime
    
    df.sort_values(by='dates',inplace=True) #sort in the ascending order of date and time to maintain natural temporal ordering
    df[features] = df[features].astype('float32')
    
    #split dateString into date and time columns
    for d in df['dates']:
        dates.append(d.date())
        times.append(d.time())

    df['time'] = times
    df['date'] = dates
    
    df['year'] = [d.year for d in df['date']]  #group by year  
    return df #df has additional columns of date, time, year now

# Does imputation in the following way:
# glucose values = Linear interpolation
# basal rates = uses the last recorded basal rate
# meals,bolus = 0 when not recorded, actual value when recorded. 
def impute_data():
    if not path.exists(data_directory + 'csv_files/'): #e.g. ../../../data/PHI/PHI_OHIO/data/csv_files/OhioT1DM-training/imputed
        os.mkdir(data_directory + 'csv_files/')
    if not path.exists(data_directory + 'csv_files/' + ohio_data)
        os.mkdir(data_directory + 'csv_files/' + ohio_data)
    for subj in subjs:
        print('-------------Imputing data for Subject: '+subj+'-------------')
        original_df = xml_to_csv(subj)
        original_df.reset_index(inplace=True)
        original_df.to_csv(data_directory + 'csv_files/' + ohio_data + '/' + subj + '.csv')
        original_df = preprocess_data(original_df)
        #original_df['dates'] = pd.to_datetime(original_df['dates'])
        #temp_basal supercedes basal
        original_df['basal'].fillna(original_df['temp_basal'],inplace=True)
        original_df['glucose_level'].loc[original_df['glucose_level'] <= threshold] = np.nan #Thresholding

        df = original_df.copy()
        #replace all missing basal rates with the last recorded basal rate
        df['basal'] = df['basal'].fillna(method='ffill')
        df = df[np.where(~df[['glucose_level']].isnull())[0][0]:] #removing leadning NANs
        #Remove rows for which cgm value is missing for more than 5 rows (roughly 25 minutes)
        mask = df.glucose_level.notna()
        a = mask.ne(mask.shift()).cumsum()
    
        df = df[(a.groupby(a).transform('size') < 5) | mask]
        
        df.reset_index(inplace=True)
        
        df.drop(['temp_basal'],axis=1,inplace=True)
        df["Time_diff"] = df['dates'].diff() #time difference between consecutive time stamps
        gaps = df[df["Time_diff"] > '00:30:00'] #indices where the time gaps are greater than 30 minutes 
        indices = gaps.index.to_series().values
        start_index = 0
        original_df = df.copy()
        #e.g. there are long gaps at indices [200,400]
        # in the first iteration get subset df[0:200], in the second iteration get df[200:400],
        # in the final iteration get df[400:]
        all_subjects_data[subj] = {}
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


            for jj in range(1,subset_df.shape[0],1):
                row = subset_df.iloc[jj]
                prev_row = subset_df.iloc[jj-1]
    
                if np.isnan(row['bolus']):
                    if ('square dual' in str(prev_row['bolus_type'])) and (not 'end' in str(prev_row['bolus_type'])):
                        subset_df['bolus'].iloc[jj] = prev_row['bolus']
                        subset_df['bolus_type'].iloc[jj] = 'square dual'
            subset_df['meal'] = subset_df['meal'].fillna(0) #fill missing meals with 0 carb
            subset_df['bolus'] = subset_df['bolus'].fillna(0) #fill missing boluses with 0
            
            subset_df['glucose_level'] = subset_df['glucose_level'].astype('float32')

            if ohio_data == 'OhioT1DM-training':
                subset_df['glucose_level'] = subset_df['glucose_level'].interpolate()[mask['glucose_level']] #imputation technique 1: linear interpolation if training set
            elif ohio_data == 'OhioT1DM-testing':
                subset_df['glucose_level'] = subset_df['glucose_level'].ffill()[mask['glucose_level']] #forward filling if test set
  
            subset_df['glucose_level'] = subset_df['glucose_level'].astype('int32')
            
            if filter_data and not ohio_data == 'OhioT1DM-testing':
                subset_df['glucose_level'] = sp.signal.medfilt(subset_df['glucose_level'].values,5) #median filtering 
        
            subset_df.dropna(subset=['basal'], inplace=True) #if basal values are missing before the first basal value is recorded

            if i < len(indices):
                df.iloc[start_index:indices[i]] = subset_df #replace the subset of data with imputed data
                start_index = indices[i] #move on to the next index
            else:
                df.iloc[start_index:] = subset_df #for last index

            if subset_df.shape[0] >= 2*history_window: #the minimum number of samples should be >= 2*history_window:
                subset_df.dropna(subset=['basal'], inplace=True)
                subset_df.reset_index(inplace=True)

                subset_df["glucose_diff"] = subset_df['glucose_level'].diff() #time difference between consecutive time stamps
                subset_df['glucose_diff'] = subset_df['glucose_diff'].fillna(0)
                all_subjects_data[subj][key] = subset_df
                key = key + 1

        if not path.exists(data_directory + 'csv_files/' + ohio_data + '/imputed'): #e.g. ../../../data/PHI/PHI_OHIO/data/csv_files/OhioT1DM-training/imputed
            os.mkdir(data_directory + 'csv_files/' + ohio_data + '/imputed')
        df["glucose_diff"] = df['glucose_level'].diff() #time difference between consecutive time stamps
        df['glucose_diff'] = df['glucose_diff'].fillna(0)
        df = df[['dates', 'glucose_level', 'basal', 'meal', 'bolus','glucose_diff']]
        df.to_csv(data_directory + 'csv_files/' + ohio_data + '/imputed/'+ subj +'.csv')
    
    with open(data_directory + 'csv_files/' + ohio_data + '/imputed/' + 'multivariate_continuous_subsets.pickle', 'wb') as f:
        # Pickle the 'all_subjects_data' dictionary using the highest protocol available.
        pickle.dump(all_subjects_data, f, pickle.HIGHEST_PROTOCOL)

# *************************************************** After XML to CSV conversion and imputation ***************************************************

# convert series to supervised learning
# Reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# if only CGM values are required (univariate series), use series = 'uni', otherwise series = 'multi'
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

def get_windowed_data():
    pickledDictionary = glob.glob(os.path.join(data_directory + 'csv_files/' + ohio_data + '/imputed/' + "multivariate_continuous_subsets.pickle"))

    with open(pickledDictionary[0], 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
    
    subjs_paths = glob.glob(data_directory + 'csv_files/' + ohio_data + '/imputed/' + '*.csv') #e.g. ../../../data/PHI/PHI_OHIO/data/csv_files/OhioT1DM-training/imputed
    subjs_paths.sort()
    all_windowed_data = {}
    for subj in subjs:   
        print('-----------------Windowing data for Subject: '+subj+'-------------------')    
        all_windowed_data[subj] = {}
        print(unpickled_data)
        keys = list(unpickled_data[subj].keys())
        reframed = pd.DataFrame()
        for key in keys:
            df = unpickled_data[subj][key]
            df = df[['dates', 'glucose_level','basal','meal', 'bolus','glucose_diff']]
            df.sort_values(by='dates',inplace=True) #sort in the ascending order of date and time to maintain natural temporal ordering
            
            #df.set_index('dates',inplace=True)
            
            df['meal'] = df['meal'].fillna(0)
            df['bolus'] = df['bolus'].fillna(0)
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
            
            reframed = pd.concat([reframed, series_to_supervised(values, history_window, prediction_window)])
        all_windowed_data[subj] = reframed

    if normalize_data:
        output_file = 'windowed_normalized_'+PH+'min.pickle'
    else:
        output_file = 'windowed_'+PH+'min.pickle'
    with open(data_directory + 'csv_files/' + ohio_data + '/imputed/'+ output_file , 'wb') as f:
        # Pickle the 'all_subjects_data' dictionary using the highest protocol available.
        pickle.dump(all_windowed_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    if len(sys.argv) > 1:
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

        print("Processing Training Data - OHIO")
        ohio_data = "OhioT1DM-training"
            impute_data()
            get_windowed_data()
        print("Processing Testing Data - OHIO")
        ohio_data = "OhioT1DM-testing"
            impute_data()
            get_windowed_data()
        
    else:
        print("Invalid input arguments")
        exit(-1)
