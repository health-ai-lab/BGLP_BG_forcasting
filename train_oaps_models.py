#!/usr/bin/env python3
# coding: utf-8
# ---------------------------------------------------------------
# This code can be used to run the following models for glucose prediction
# for OAPS or OhioT1DM data:

# Linear Regression
# Random Forest Regression Trees
# Support vector regression
# Ensemble
# single_step/multi_output LSTM
# RNN

# Author: Hadia Hameed
# References:
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
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
import pytz

#machine learning packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#miscellaneous
import math
from joblib import dump, load
import json
import random
import pickle

#set which GPU to use
#Can be any number through 0 to 7
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[-1]

#(default configuration for model)
state_vector_length = 32 #after manual grid search
epochs = 100
batch_size = 248
activation = 'relu' #activation for LSTM and RNN
poly_degree = 2 #degree of polynomial

models = ['REG','SVR','TREE','ENSEMBLE','LSTM', 'RNN']

if sys.argv[-1] == '0':
    #seg = 'filtered_false/'
    seg = 'unfiltered_imputed/' #No filtering but imputing missing CGM values
elif sys.argv[-1] == '1':
    #seg = 'filtered_true/'
    seg = 'filtered_imputed/' 
elif sys.argv[-1] == '2':
    #seg = 'filtered_false_extrapolate/'
    seg = 'unfiltered_unimputed/' #No filtering and not imputing missing CGM values
elif sys.argv[-1] == '3':
    #seg = 'filtered_true_extrapolate/'
    seg = 'filtered_unimputed/'

# unpickle a pickled dictionary
def unpickle_data(data_path):
    with open(data_path, 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
    return unpickled_data

# re-constructs data based on single-step/multi-output settings
# returns X, y as "features" (historical data) and "labels" (future data)
def process_data(df):
    excluded_keys = list()
    for key in df.keys():
        if 'date' in key:
            excluded_keys.append(key)
    df.drop(excluded_keys, axis=1, inplace=True)
    
    if prediction_type == 'single':
        df.drop(df.columns[-prediction_window:-1], axis=1, inplace=True) #gets a single reading 30 minutes into the future
    
    
    if dimension == 'univariate': #if univariate, drop all columns that do not have CGM values
        excluded_keys = list()
        for key in df.keys():
            if 'CGM' not in key:
                excluded_keys.append(key)
        df.drop(excluded_keys, axis=1, inplace=True)

    data = df.values
    data = data.astype('float32')

    if prediction_type == 'single':
        X, y = data[:, :-1], data[:, -1:] #x(t+5)
    elif prediction_type == 'multi':
        X, y = data[:, :-prediction_window], data[:, -prediction_window:] #x(t), x(t+1), ... , x(t+5)
        
    return X , y


#initializes a deep learning model (LSTM or RNN)
def deepLearningModels(model_name,X,y):
    model = Sequential()
    if model_name == 'LSTM':
        model.add(LSTM(state_vector_length, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        
    elif model_name == 'RNN':
        model.add(SimpleRNN(state_vector_length, activation=activation, input_shape=(X.shape[1], X.shape[2])))

    prediction_window = y.shape[1]
    model.add(Dense(prediction_window))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    return model


#initializes a baseline model (Linear regression, 2nd order polynomial regression, Random forest regressor or ensemble of all three)
def baselineModels(model_name):
    if model_name == 'REG':
        model = LinearRegression()
    elif model_name == 'SVR':
        model = SVR(cache_size=1000000)
    elif model_name == 'TREE':
        model = RandomForestRegressor()
    elif model_name == 'ENSEMBLE': #ensemble of linear, polynomial regression, Random Forest Regressor
        model = [] #list of models
        model.append(LinearRegression())
        model.append(SVR())
        model.append(RandomForestRegressor())
        
    if prediction_type == 'multi':
        model = MultiOutputRegressor(model, n_jobs=-1)
    
    return model


#******************************* MAIN Function for CGM prediction ***************************************
def CGM_Prediction(model_name,unpickled_train_data ,unpickled_test_data,ij ):

    train_subjs = list(unpickled_train_data.keys())
    test_subjs = list(unpickled_test_data.keys())
    random.shuffle(train_subjs)

    #nested function to test the model trained below
    def test_model(model,df):
        test_X, test_y = process_data(df) #get features and labels for test data
        if model_type == 'deep':
            #[no. of samples, timestamps, no. of features]
            test_X = test_X.reshape((test_X.shape[0], history_window , n_features))
        if not model_name == 'ENSEMBLE':
            y_bar = model.predict(test_X)
        elif model_name == 'ENSEMBLE':
            weight = 2 #more weight given to linear regression and Tree
            y_bar_linear = model[0].predict(test_X)
            y_bar_svr = model[1].predict(test_X)
            y_bar_tree = model[2].predict(test_X)
            y_bar = [(weight*g + weight*h + i) / (2*weight+1) for g, h, i in zip(y_bar_tree, y_bar_linear,y_bar_svr)]
        
        if prediction_type == 'multi':
            for ii in range(len(y_bar)):
                y_bar[ii] = [int(element) for element in y_bar[ii]]
        else:
            y_bar = [int(element) for element in y_bar]

        predicted_values = pd.DataFrame(list(zip(test_y,y_bar)),columns=['True','Estimated'])

        #for multi-output forecasting, calculate RMSE using the last value in the sequence
        if prediction_type == 'multi':
            y_bar = [last for *_, last in y_bar]
            test_y = [last for *_, last in test_y]
        
        testScore = math.sqrt(mean_squared_error(y_bar, test_y))
        return testScore, predicted_values

    def train_model(model,train_X,train_y):
        n_train = int(0.7*train_X.shape[0])
        
        if model_type == 'deep':
            train_X, val_X = train_X[:n_train, :,:], train_X[n_train:, :,:]
            train_y, val_y = train_y[:n_train], train_y[n_train:]
            early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
            history = model.fit(train_X, train_y, validation_data=(val_X, val_y), shuffle=False, epochs=epochs,batch_size=batch_size,verbose=0,callbacks=[early_stop])
            y_bar = model.predict(val_X)
        
        elif model_type == 'baseline':
            train_X, val_X = train_X[:n_train,:], train_X[n_train:,:]
            train_y, val_y = train_y[:n_train], train_y[n_train:]
            if not model_name == 'ENSEMBLE':
                model.fit(train_X, train_y) #training the algorithm 
                y_bar = model.predict(val_X)
                history = {}  
        
            elif model_name == 'ENSEMBLE':
                model[0].fit(train_X, train_y) #linear regression
                model[1].fit(train_X, train_y) #polynomial regression
                model[2].fit(train_X, train_y) #random forest regression tree
                history = {}
                weight = 2 #more weight given to linear regression and Tree
                y_bar_linear = model[0].predict(val_X)
                y_bar_svr = model[1].predict(val_X)
                y_bar_tree = model[2].predict(val_X)
                y_bar = [(weight*g + weight*h + i) / (2*weight+1) for g, h, i in zip(y_bar_tree, y_bar_linear,y_bar_svr)]
        
        if prediction_type == 'multi':
            for ii in range(len(y_bar)):
                y_bar[ii] = [int(element) for element in y_bar[ii]]
        else:
            y_bar = [int(element) for element in y_bar]

        #for multi-output forecasting, calculate RMSE using the last value in the sequence
        if prediction_type == 'multi':
            y_bar = [last for *_, last in y_bar]
            val_y = [last for *_, last in val_y]

        valScore = math.sqrt(mean_squared_error(y_bar, val_y))
        print('Validation RMSE: %.3f' % valScore) 
        return model, history, valScore

    testScores = list()
    valScores = list() #validation error
    subjects = list()
    model_train_history = list()
    model_val_history = list()
    counter = 0 #keeping track of when the model was initialized

    for subj in train_subjs:
        print('----------Training on subject: ',subj,'----------')
        df = unpickled_train_data[subj].copy()
        train_X, train_y = process_data(df)

        #[no. of samples, timestamps, no. of features]
        if model_type == 'deep':
            train_X = train_X.reshape((train_X.shape[0], history_window , n_features))

        if counter == 0:
            if model_type == 'deep':
                model = deepLearningModels(model_name,train_X, train_y)
            elif model_type == 'baseline':
                model = baselineModels(model_name)
        counter = counter + 1

        model,history,valScore = train_model(model,train_X,train_y)

        if model_type == 'deep':
            model_train_history.extend(history.history['mean_squared_error'])
            model_val_history.extend(history.history['mean_squared_error'])

        valScores.append(valScore)
        subjects.append(subj)

    all_subjs_predicted_values = {}
    for subj in subjects:
        print('----------Testing on subject: ',subj,'----------')
        df = unpickled_test_data[subj].copy()
        testScore, predicted_values = test_model(model,df)
        all_subjs_predicted_values[subj] = predicted_values 
        print('Test RMSE: %.3f' % testScore) 
        testScores.append(testScore)              

    results_df = pd.DataFrame(list(zip(subjects,valScores,testScores)),columns=['Subject','valRMSE','testRMSE'])
    results_df.sort_values(by=['Subject'], inplace = True)      
    return results_df, model, model_train_history,model_val_history,all_subjs_predicted_values

def make_directories():
    #datetime_now = datetime.datetime.now()
    #datetime_now = datetime_now.strftime("%d-%m-%Y_%I-%M-%S_%p")
    if not path.exists(output_directory):
        os.mkdir(output_directory)
    if not path.exists(output_directory + 'overall_results'):
        os.mkdir(output_directory + 'overall_results')
    if not path.exists(output_directory + 'overall_results' + '/' + model_name):
        os.mkdir(output_directory + 'overall_results' + '/' + model_name)
    if not path.exists(output_directory + 'overall_results' + '/' + model_name + '/' + prediction_type + '_' + dimension + '_' + PH):
        os.mkdir(output_directory + 'overall_results' + '/' + model_name + '/' + prediction_type + '_' + dimension + '_' + PH)

    return output_directory + 'overall_results' + '/' + model_name + '/' + prediction_type + '_' + dimension + '_' + PH
    
  
def main(model_name):
    overall_results = pd.DataFrame() #saving subject ID, test and validation RMSE for each iteration, overall mean RMSE and MAE 
    results_directory = make_directories()
    no_iterations = range(5)

    if normalize_data:
        substring = 'normalized_'+PH+'min'
    else:
        substring = PH+'min'

    if dataset == 'oaps':
        print('Getting data from ', data_directory + seg + '\n')
        unpickled_train_data = unpickle_data(data_directory + seg + 'windowed_train_' + substring + '.pickle') #e.g. windowed_train_normalized_60min.pickle
        unpickled_test_data = unpickle_data(data_directory + seg + 'windowed_test_' + substring + '.pickle') 
    elif dataset == 'ohio':
        unpickled_train_data = unpickle_data(data_directory + 'OhioT1DM-training/imputed/'+'windowed_' + substring + '.pickle') #e.g. windowed_normalized_60min.pickle
        unpickled_test_data = unpickle_data(data_directory + 'OhioT1DM-testing/imputed/'+'windowed_' + substring + '.pickle')

    for i in no_iterations:
        print('Iteration #: ',i)
        results_df, model, model_train_history,model_val_history, all_subjs_predicted_values = CGM_Prediction(model_name,unpickled_train_data,unpickled_test_data,i )
        if i == no_iterations[0]:
            overall_results = pd.concat([results_df, overall_results], axis=1)
        else:
            overall_results = results_df.merge(overall_results, on='Subject', how='inner', suffixes=('_1', '_2'))
        if not path.exists(results_directory  + '/' + seg):
            os.mkdir(results_directory  + '/' + seg)
        if not path.exists(results_directory  + '/' + seg + str(i)):
            os.mkdir(results_directory + '/' + seg + str(i))
    
        filename = results_directory  + '/' + seg + str(i) + '/' + prediction_type + '_' + dimension + '_' + substring
        
        if save_results:
            with open(results_directory  + '/' + seg + str(i) + '/' + 'predicted_values.pickle', 'wb') as f:
                pickle.dump(all_subjs_predicted_values , f, pickle.HIGHEST_PROTOCOL)
            overall_results.to_csv(filename + '.csv')
            if model_type == 'deep':
                model.save(filename + '.h5')
            else:
                dump(model, filename+'.joblib') 
            if model_type == 'deep':
                model_history = {}
                model_history['train'] = model_train_history
                model_history['val'] = model_val_history
                with open(filename+'_history.pickle', 'wb') as f:
                    pickle.dump(model_history , f, pickle.HIGHEST_PROTOCOL)
    
    if save_results:
        exclude_keys = list()
        for key in overall_results.keys():
            if 'testRMSE' not in key:
                exclude_keys.append(key)

        test_rmse_df = overall_results.drop(exclude_keys, axis=1)
        overall_results['Mean Test RMSE']= test_rmse_df.mean(axis=1) 
        overall_results['STD Test RMSE']= test_rmse_df.std(axis=1) 

        exclude_keys = list()
        for key in overall_results.keys():
            if 'valRMSE' not in key:
                exclude_keys.append(key)

        val_rmse_df = overall_results.drop(exclude_keys, axis=1)
        overall_results['Mean Val RMSE']= val_rmse_df.mean(axis=1) 
        overall_results['STD Val RMSE']= val_rmse_df.std(axis=1) 

        filename = results_directory + '/' + seg + prediction_type + '_' + dimension + '_' + substring
        overall_results.to_csv(filename+'.csv')
       
if __name__ == "__main__":
    if len(sys.argv) > 4:
        root_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
        history_window = int(sys.argv[4]) #12
        prediction_window = int(sys.argv[5]) #30 or 60 minutes
        dimension = sys.argv[6] #univariate or multivariate
        prediction_type = sys.argv[7] #single or multi (single-step or multi-output)
        if sys.argv[8] == 'False':
            normalize_data = False
        else:
            normalize_data = True
        model_name = sys.argv[9]
        dataset = sys.argv[10]
        if sys.argv[11] == 'False':
            save_results = False
        else:
            save_results = True

        PH = str(prediction_window) #prediction horizon
        if prediction_window == 30 or prediction_window == 60:
            prediction_window = prediction_window//5

        if (model_name == 'RNN') or (model_name == 'LSTM'):
            model_type = 'deep'

        elif model_name == 'REG' or model_name == 'TREE' or model_name == 'SVR' or model_name == 'ENSEMBLE':
            model_type = 'baseline'
        else:
            print('Model not found. Please choose model name from [RNN, LSTM, Reg, Tree, SVR, Ensemble]')
            exit(-1)
        
        if dimension == 'univariate':
            n_features = 1
        elif dimension == 'multivariate':
            n_features = 5
        print("Starting experiments for the following settings: prediction_window: "+PH+"; dimension: "+dimension+"; prediction_type: "+prediction_type+"; normalize_data: "+str(normalize_data)+"; model_name: "+model_name+" ;dataset: "+dataset+"; save_results: "+str(save_results)+'; ablation code: '+seg+'\n')
        main(model_name)
        print("Experiments completed for the following settings: prediction_window: "+PH+"; dimension: "+dimension+"; prediction_type: "+prediction_type+"; normalize_data: "+str(normalize_data)+"; model_name: "+model_name+" ;dataset: "+dataset+"; save_results: "+str(save_results)+'; ablation code: '+seg+'\n')
    else:
        print("Invalid input arguments")
        exit(-1)
