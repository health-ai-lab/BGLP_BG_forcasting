#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------------------

# Author: Hadia Hameed
# References:
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://github.com/suetAndTie/ClarkeErrorGrid/blob/master/ClarkeErrorGrid.py
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
from datetime import timedelta
import time

#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#machine learning packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from joblib import dump, load
from operator import add

#miscellaneous
import math
import random
import pickle
import xml.etree.ElementTree as ET


# This function gets the test set from multivariate continuous subsets
# and windows it
def get_test_data():

    #for windowing
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            
        output_features = ['date','glucose_level']
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (j, i)) for j in input_features]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (j)) for j in input_features]
            else:
                names += [('%s(t+%d)' % (j, i)) for j in input_features]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    with open(data_directory+'csv_files/OhioT1DM-testing/imputed/multivariate_continuous_subsets.pickle', 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
         
    with open(data_directory+'csv_files/OhioT1DM-training/imputed/multivariate_continuous_subsets.pickle', 'rb') as f:
        unpickled_train_data = pickle.load(f, encoding='latin1')
    
    data_dictionary = {}
    for subj in unpickled_data.keys():
        print('-----------------Windowing data for Subject: '+subj+'-------------------')    
        keys = list(unpickled_data[subj].keys()) #subsets of continuous time-series data for each subject
        windowed_data = pd.DataFrame()
        #getting raw data for the subject
        xml_doc = ET.parse(data_directory+'OhioT1DM-testing/'+subj+'-ws-testing.xml')
        root = xml_doc.getroot() 
        dates = list()
        values = list()
        for row in root[0]: #glucose_level tag
            signal = row.attrib
            signal_keys = list(signal.keys()) #[<ts><value>]
            dates.append(datetime.datetime.strptime(signal['ts'],'%d-%m-%Y %H:%M:%S')) #[<ts>]
            values.append(signal['value']) #[<value>]

        CGM_data_original = pd.DataFrame(columns={'dates','glucose_level'})
        CGM_data_original['dates'] = dates
        CGM_data_original['glucose_level'] = values
        CGM_data_original.sort_values(by='dates',inplace=True) #sort in the ascending order of date and time to maintain natural temporal ordering
    
        for i in range(len(keys)-1):
            df1 = unpickled_data[subj][keys[i]]
            if i == 0:
                df_train = unpickled_train_data[subj][list(unpickled_train_data[subj].keys())[-1]].iloc[-2*history_window:]
                df_train['dates'] = pd.to_datetime(df_train['dates'])
                df1 = pd.concat([df_train,df1],ignore_index=True)
            df2 = unpickled_data[subj][keys[i+1]]
            df1['dates'] = pd.to_datetime(df1['dates'])
            df2['dates'] = pd.to_datetime(df2['dates'])
            df1.sort_values(by='dates',inplace=True) #sort in the ascending order of date and time to maintain natural temporal ordering
            df2.sort_values(by='dates',inplace=True)
            df1 = df1[input_features]
            df2 = df2[input_features]
            
            d1 = df1[['dates']].iloc[-1].values
            start = pd.to_datetime(d1[0])
            d2 = df2[['dates']].iloc[0].values
            end = pd.to_datetime(d2[0])
            seconds = (end - start).total_seconds()
            step = timedelta(minutes=5)
            array = []
            for j in range(300, int(seconds), int(step.total_seconds())):
                array.append(start + timedelta(seconds=j))
            array = pd.to_datetime(array)
            bg_array = np.empty((array.shape))
            bg_array[:] = np.nan
            intermediate_df = pd.DataFrame(columns={'dates','glucose_level'})
            intermediate_df['dates'] = array
            intermediate_df['glucose_level'] = bg_array
            df1 = df1.append(intermediate_df, ignore_index=True)
            df1 = df1.interpolate()
            df2 = pd.concat([df1.iloc[-(history_window+prediction_window+1):], df2]).reset_index(drop = True) 

            if i == 0:
                windowed_data = pd.concat([windowed_data, series_to_supervised(df1.values, history_window, prediction_window)]) 
            windowed_data = pd.concat([windowed_data, series_to_supervised(df2.values, history_window, prediction_window)])

        dates = windowed_data.filter(regex='^date',axis=1).values #extract timestamps
        windowed_data = windowed_data.loc[:,~windowed_data.columns.str.startswith('date')] #remove dates column

        if dimension == 'univariate': #if univariate, drop all columns that do not have CGM values
            windowed_data = windowed_data.loc[:,windowed_data.columns.str.startswith('glucose_level')] 
        
        #get past values as feature vector "X"
        exclude_columns = list()
        for key in windowed_data.keys():
            if 't-' not in key:
                exclude_columns.append(key)
        X = windowed_data.drop(exclude_columns, axis=1)
        X = X.values
        X = X.astype('float32')

        #get future values as feature vector "y"
        exclude_columns = list()
        for key in windowed_data.keys():
            if 'glucose_level(t+'+str(prediction_window-1) not in key and 'date(t+'+str(prediction_window-1) not in key:
                exclude_columns.append(key)
        y = windowed_data.drop(exclude_columns, axis=1)
        y = y.values
        y = y.astype('float32')

        data_dictionary[subj] = {} 
        data_dictionary[subj]['X'] = X #past BG values
        data_dictionary[subj]['y'] = y #future BG values
        data_dictionary[subj]['dates'] = dates

    return data_dictionary

#reference: https://github.com/suetAndTie/ClarkeErrorGrid/blob/master/ClarkeErrorGrid.py
#This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
#of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
def clarke_error_grid(ref_values, pred_values, title_string,mycolor):

    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
        #ref_values[ref_values.index(max(ref_values))]=400-1
        #pred_values[pred_values.index(max(pred_values))]=400-1
    if min(ref_values) < 0 or min(pred_values) < 0:
        print("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))
    
    
    #Clear plot
    #plt.clf()

    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color=mycolor, s=8)
    plt.title("Clarke Error Grid for "+title_string)
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))

    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    #Add zone titles
    plt.text(30, 15, "A", color = 'red', fontsize=15)
    plt.text(370, 260, "B", color = 'red',fontsize=15)
    plt.text(280, 370, "B", color = 'red',fontsize=15)
    plt.text(160, 370, "C", color = 'red',fontsize=15)
    plt.text(160, 15, "C", color = 'red',fontsize=15)
    plt.text(30, 140, "D", color = 'red',fontsize=15)
    plt.text(370, 120, "D", color = 'red',fontsize=15)
    plt.text(30, 370, "E", color = 'red',fontsize=15)
    plt.text(370, 15, "E", color = 'red',fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    return plt, zone
 
if __name__ == "__main__":
    data_directory = sys.argv[1]
    output_directory = sys.argv[2]
    pipeline = sys.argv[3] #learning pipeline [student,teacher,retrain,teacher_student]
    dimension = 'univariate' #test extrapolation BGLP rule was only tested for univariate setting
    PH = sys.argv[4] #prediction horizon (minutes) [30, 60]
    model_name = sys.argv[5] #[LSTM or RNN]
    history_window = int(sys.argv[6]) #12 past samples (an hour of data)
    prediction_window = int(PH)//5 
    model_directory = sys.argv[7]

    if dimension == 'univariate':
        input_features = ['dates','glucose_level']
        
    elif dimension == 'multivariate':
        input_features = ['dates','glucose_level','basal','meal','bolus','glucose_diff']
    
    no_of_features = len(input_features) - 1 #univariate (1), multivariate (5)
    configuration = 'single_'+dimension #single_univariate or single_multivariate

    if not path.exists(output_directory):
        os.mkdir(output_directory)
    results_directory = output_directory+PH+'_min/'
    if not path.exists(results_directory):
        os.mkdir(results_directory)
    model_directory = model_directory+pipeline+'/'+configuration+'_'+PH+'/'+model_name+'/' #e.g. 'OHIO_models/student/single_univariate_30/RNN/'
    test_subjs = ['540','544','552','567','584','596'] #[(new) 540,544,552,567,584,596] for test set
    
    folders = list(os.walk(model_directory))[0][1] #different RNN model trained over n iterations (n = 10)

    test_data = get_test_data() #get windowed data

    results = {}
    rmse = list() 
    mae = list()
    fig = plt.figure(figsize=(10,10)) #for plotting Clarke's Error Grid

    for subject in test_subjs:
        results[subject] = {}
        print('\n---------Testing on subject ',subject,'---------\n')
        y_bars = pd.DataFrame()
        X = test_data[subject]['X'] #past values
        y = test_data[subject]['y'] # true future values
        #[no. of samples, timestamps, no. of features]
        X = X.reshape((X.shape[0], history_window , no_of_features)) 
        for folder in folders:
            print('---------Using model #',folder,'---------')
            model = load_model(model_directory+folder+'/'+configuration+'_'+model_name+'_'+PH+'min.h5')
            y_bar = model.predict(X)
            y_bar = [int(element) for element in y_bar]
            y_bars[folder] = y_bar
        
        y_bars['overall'] = y_bars.mean(axis=1)  #taking mean of glucose values predicted by the different models
        y_bar = y_bars['overall'].values
        
        y = [int(element) for element in y]
        y_bar = [int(element) for element in y_bar]
        dates = [last for *_, last in test_data[subject]['dates']] #get the last date in the sequence
        predicted_values = pd.DataFrame(list(zip(dates,y,y_bar)),columns=['dates','True','Estimated'])

        #getting raw data for the subject
        xml_doc = ET.parse(data_directory+'OhioT1DM-testing/'+subject+'-ws-testing.xml')
        root = xml_doc.getroot() 
        dates = list()
        values = list()
        for row in root[0]: #glucose_level tag
            signal = row.attrib
            signal_keys = list(signal.keys()) #[<ts><value>]
            dates.append(datetime.datetime.strptime(signal['ts'],'%d-%m-%Y %H:%M:%S')) #[<ts>]
            values.append(signal['value']) #[<value>]

        CGM_data_original = pd.DataFrame(columns={'dates','glucose_level'})
        CGM_data_original['dates'] = dates
        CGM_data_original['glucose_level'] = values
        n = history_window
        CGM_data_original = CGM_data_original.iloc[n:]
        predicted_values['dates'] = pd.to_datetime(predicted_values['dates'])
        
        #inner join of raw and predcited BG value tables
        merged_df = CGM_data_original.merge(predicted_values, on='dates', how='inner', suffixes=('_1', '_2'))
        merged_df = merged_df[['dates','True','Estimated']]
        results[subject]['predicted_values'] = merged_df
        
        #saving true and estimated values to text file
        dc_ID = subject
        filename = dc_ID + '_'+PH
        merged_df.to_csv(results_directory+filename+'.txt',  index=True, sep=' ', mode='w')

        #calculating RMSE and MAE
        y = results[subject]['predicted_values']['True'].values
        y_bar = results[subject]['predicted_values']['Estimated'].values
        rmse_score = math.sqrt(mean_squared_error(y_bar, y))
        mae_score = mean_absolute_error(y, y_bar)
        rmse.append(rmse_score)
        mae.append(mae_score)
        results[subject]['RMSE'] = rmse_score
        results[subject]['MAE'] = mae_score

        #plotting CEG for all the subjects on the same plot
        plot, _ = clarke_error_grid(y, y_bar, 'PH = '+PH+' minutes','black')

    #saving CEG plot
    plt.savefig(results_directory+'error_grid_plot.png')

    #saving results dictionary. Contains RMSE, MAE, predcited_values dataframe for each subject
    with open(results_directory+'results_dictionary.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)    

    #results summary. RMSE, MAE for each subject. Mean RMSE and MAE for all the 6 subjects
    overall_results = pd.DataFrame(list(zip(test_subjs,rmse,mae)),columns=['Subject ID','RMSE','MAE'])
    total_mean = overall_results.mean(axis=0)
    df2 = pd.DataFrame([total_mean])
    df3 = pd.DataFrame([None]*len(overall_results.columns)).T
    df3.columns = overall_results.columns
    overall_results = overall_results.append(df3)
    overall_results = overall_results.append(df2)
    overall_results['Subject ID'].iloc[-1] = None
    overall_results.to_csv(results_directory+'results_summary.csv')
