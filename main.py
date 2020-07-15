#!/usr/bin/env python
# coding: utf-8
# ---------------------------------------------------------------
# This code can be used to do blood glucose prediction using 
# OhioT1DM dataset for four different learning pipelines
# ----------------------
# Approach I = 'Student' 
# Train an LSTM using OhioT1DM data only
# ----------------------
# Approach II = 'Teacher without re-training'
# Use an LSTM pre-trained on OAPS data without any further 
# re-training on OhioT1DM dataset to test on OhioT1DM test set
# ----------------------
# Approach III = 'Teacher with re-training'
# Use an LSTM pre-trained on OAPS data and train it further 
# on OhioT1DM data and then test on OhioT1DM test set
# ----------------------
# Approach IV = 'Teacher-student'
# Use an LSTM pre-trained on OAPS data to make soft-estimations
# of the future glucose values. Use these as target vectors to 
# train an RNN model on OhioT1DM data instead of the actual
# ground truth.

# Author: Hadia Hameed
# References:
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
# https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html
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

#plotting packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#datetime packages
import datetime
import time

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

#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[-1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #mute tensorflow warnings

#(default configuration for model)
state_vector_length = 32 #after manual grid search
epochs = 1000
batch_size = 128
activation = 'relu' #activation for LSTM and RNN

#(default configuration for mimic learning) 
mimic_pipelines =  ['student','teacher','retrain','teacher_student']
mimic_pipeline = mimic_pipelines[1] #default = student

# unpickle a pickled dictionary
def unpickle_data(data_path):
    with open(data_path, 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
    return unpickled_data

# re-constructs data based on single-step/multi-output settings
# returns X, y as "features" (historical data) and "labels" (future data)
def process_data(df):
    dates = df.filter(regex='^date',axis=1).values #extract datestamps
    df = df.loc[:,~df.columns.str.startswith('date')] #remove dates column

    if prediction_type == 'single':
        df.drop(df.columns[-prediction_window:-1], axis=1, inplace=True) #gets a single reading 30 minutes into the future
    if dimension == 'univariate': #if univariate, drop all columns that do not have CGM values
        df = df.loc[:,df.columns.str.startswith('CGM')] #remove dates column
    
    data = df.values
    data = data.astype('float32')

    if prediction_type == 'single':
        X, y = data[:, :-1], data[:, -1:] #x(t+5)
    elif prediction_type == 'multi':
        X, y = data[:, :-prediction_window], data[:, -prediction_window:] #x(t), x(t+1), ... , x(t+5)
     
    return X , y, dates

#initializes a deep learning model (LSTM or RNN)
def deepLearningModels(model_name,X,y):
    model = Sequential()
    if model_name == 'LSTM':
        model.add(LSTM(state_vector_length, activation=activation, return_sequences = False, input_shape=(X.shape[1], X.shape[2]))) #hidden layer
    elif model_name == 'RNN': 
        model.add(SimpleRNN(state_vector_length, activation=activation, input_shape=(X.shape[1], X.shape[2])))
    elif model_name == 'ANN': 
        model.add(Dense(state_vector_length, input_dim=X.shape[1], activation=activation)) #hidden layer
    prediction_window = y.shape[1]
    model.add(Dense(prediction_window)) #output layer
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

#******************************* MAIN Function for CGM prediction ***************************************
def CGM_Prediction(model_name,iteration):
    if normalize_data:
        substring = 'normalized_'+PH+'min'
    else:
        substring = PH+'min'
        
    unpickled_train_data = unpickle_data(data_directory + 'OhioT1DM-training/imputed/'+'windowed_' + substring + '.pickle')
    unpickled_test_data = unpickle_data(data_directory + 'OhioT1DM-testing/imputed/'+'windowed_' + substring + '.pickle')
    
    train_subjs = list(unpickled_train_data.keys()) #[(old): 559,563,570,575,588,591, (new) 540,544,552,567,584,596]
    val_subjs = ['559','563','570','575','588','591'] #[(old): 559,563,570,575,588,591]
    test_subjs = ['540','544','552','567','584','596'] #[(new) 540,544,552,567,584,596]
    random.shuffle(train_subjs)

    #nested function to test the model trained below
    def test_model(student_model,teacher_model,test_X,test_y, dates):
        y_bar = student_model.predict(test_X) #predictions for pipeline I-III
        if mimic_pipeline == 'teacher_student': #predictions for pipeline IV
            y_bar_teacher = teacher_model.predict(test_X.reshape((test_X.shape[0], history_window , n_features)))
            y_bar = list( map(add, y_bar, y_bar_teacher) )
            y_bar[:] = [x / 2 for x in y_bar]

        #single-step forecast
        if prediction_type == 'single':
            y_bar = [int(element) for element in y_bar]
            test_y = [int(element) for element in test_y]
        #multi-output forecast
        else:
            for ii in range(len(y_bar)):
                y_bar[ii] = [int(element) for element in y_bar[ii]]
                test_y[ii] = [int(element) for element in test_y[ii]]
        #for multi-output forecasting, calculate RMSE using the last value in the sequence
        if prediction_type == 'multi':
            y_bar = [last for *_, last in y_bar]
            test_y = [last for *_, last in test_y]
            
        dates = [last for *_, last in dates]
        predicted_values = pd.DataFrame(list(zip(dates,test_y,y_bar)),columns=['Dates','True','Estimated'])
        testScore = list()
        testScore.append(math.sqrt(mean_squared_error(y_bar, test_y))) #RMSE
        testScore.append(mean_absolute_error(test_y, y_bar)) #MAE
        return testScore, predicted_values

    def train_model(model,train_X,train_y,val_X,val_y):
        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1) 
        history = model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=epochs,batch_size=batch_size,verbose=0,callbacks=[early_stop])
        return model, history

    testScores_RMSE = list()
    testScores_MAE = list()
    subjects = list()
    model_train_history = list()
    model_val_history = list()
    counter = 0
    
    #load teacher model
    if prediction_type == 'multi' and dimension == 'multivariate':
        model_path = model_directory+model_name+"/multi_multivariate_"+PH+"/"+iteration+"/multi_multivariate_"+PH+"min.h5"
    elif prediction_type == 'multi' and dimension == 'univariate':
        model_path = model_directory+model_name+"/multi_univariate_"+PH+"/"+iteration+"/multi_univariate_"+PH+"min.h5"
    elif prediction_type == 'single' and dimension == 'univariate':
        model_path = model_directory+model_name+"/single_univariate_"+PH+"/"+iteration+"/single_univariate_"+PH+"min.h5"
    elif prediction_type == 'single' and dimension == 'multivariate':
        model_path = model_directory+model_name+"/single_multivariate_"+PH+"/"+iteration+"/single_multivariate_"+PH+"min.h5"
    
    teacher_model = load_model(model_path)

    if mimic_pipeline == 'teacher_student':
        model_name = 'ANN'
    if not mimic_pipeline == 'teacher': #no training required for teacher
        counter = 0
        for train_subj in train_subjs:
            print('----------Getting training data for subject: ',train_subj,'----------')
            train_data = unpickled_train_data[train_subj]
            X, y, _ = process_data(train_data)

            if mimic_pipeline == 'teacher_student': #if the learning pipeline is Approach IV student model with soft estimations from teacher model
                y = teacher_model.predict(X.reshape((X.shape[0], history_window , n_features)))
            if counter == 0:
                train_X = X
                train_y = y
            else:
                train_X = np.concatenate((train_X, X), axis=0)
                train_y = np.concatenate((train_y, y), axis=0)
            counter = counter + 1
            
        counter = 0
        for val_subj in val_subjs:
            print('----------Getting validation data for subject: ',val_subj,'----------')
            val_data = unpickled_test_data[val_subj]
            X, y, _ = process_data(val_data)
    
            if counter == 0:
                val_X = X
                val_y = y
            else:
                val_X = np.concatenate((val_X, X), axis=0)
                val_y = np.concatenate((val_y, y), axis=0)
            counter = counter + 1

        if not mimic_pipeline == 'teacher_student': #no need to reshape data for ANN which is the student model in pipeline IV
            train_X = train_X.reshape((train_X.shape[0], history_window , n_features))
            val_X = val_X.reshape((val_X.shape[0], history_window , n_features))

        if mimic_pipeline == 'retrain':
            student_model = teacher_model #if the learning pipeline is Approach III, retrain the teacher model

        else: 
            student_model = deepLearningModels(model_name,train_X, train_y) #if the learning pipeline is Approach I, use student model without any teacher model
        
        student_model, history = train_model(student_model, train_X, train_y, val_X, val_y)
        model_train_history = history.history['mean_squared_error']
        model_val_history = history.history['val_mean_squared_error']
        
    else:
        student_model = teacher_model #if the learning pipeline is Approach II teacher without re-training

    all_predicted_values = {}
    #Testing on test subjects
    for subj in test_subjs:
        print('----------Testing on subject: ',subj,'----------')
        df = unpickled_test_data[subj]
        test_X, test_y, dates = process_data(df) #get features and labels for test data
        if not mimic_pipeline == 'teacher_student':
            #[no. of samples, timestamps, no. of features]
            test_X = test_X.reshape((test_X.shape[0], history_window , n_features))

        testScore, predicted_values = test_model(student_model,teacher_model,test_X,test_y, dates)
        testScores_RMSE.append(testScore[0])
        testScores_MAE.append(testScore[1])
        print('Test RMSE: %.3f' % testScore[0]) 
        print('Test MAE: %.3f' % testScore[1]) 
        subjects.append(subj)
        all_predicted_values[subj] = predicted_values
       


    results_df = pd.DataFrame(list(zip(subjects,testScores_RMSE,testScores_MAE)),columns=['Subject','RMSE','MAE'])
    results_df.sort_values(by=['Subject'], inplace = True)          
    return results_df, student_model, model_train_history,model_val_history, all_predicted_values
    
def main(model_name):
    
    overall_results = pd.DataFrame()

    if mimic_pipeline == 'teacher':
        no_iterations = 3
    else:
        no_iterations = 10
    
    if save_results:
        if not path.exists(output_directory):
            os.mkdir(output_directory)
        if not path.exists(output_directory+mimic_pipeline): 
            os.mkdir(out+mimic_pipeline)
        if not path.exists(output_directory+mimic_pipeline+'/'+prediction_type+'_'+dimension+'_'+PH): 
            os.mkdir(output_directory+mimic_pipeline+'/'+prediction_type+'_'+dimension+'_'+PH)
        if not path.exists(output_directory+mimic_pipeline+'/'+prediction_type+'_'+dimension+'_'+PH+'/'+model_name): 
            os.mkdir(output_directory+mimic_pipeline+'/'+prediction_type+'_'+dimension+'_'+PH+'/'+model_name)

    if normalize_data:
        substring = '_normalized_'+PH+'min'
    else:
        substring = '_'+PH+'min'

    for i in range(no_iterations):
        print('Iteration #: ',i)
        if mimic_pipeline == 'teacher':
            results_df, model, model_train_history,model_val_history,all_predicted_values = CGM_Prediction(model_name,str(i))
        else:
            results_df, model, model_train_history,model_val_history,all_predicted_values = CGM_Prediction(model_name,'0')
        if i == 0:
            overall_results = pd.concat([results_df, overall_results], axis=1)
        else:
            overall_results = results_df.merge(overall_results, on='Subject', how='inner', suffixes=('_1', '_2'))
        
        if save_results:
            results_directory = output_directory+mimic_pipeline+'/'+prediction_type+'_'+dimension+'_'+PH+'/'+model_name+'/'+str(i)
            if not path.exists(results_directory):
                os.mkdir(results_directory)
            with open(results_directory+'/predicted_values.pickle', 'wb') as f:
                # save model history
                pickle.dump(all_predicted_values, f, pickle.HIGHEST_PROTOCOL)
            model_history = {}
            model_history['train'] = model_train_history
            model_history['val'] = model_val_history
            #save model
            filename = results_directory+'/%s_%s_%s_%smin' %(prediction_type,dimension,model_name,PH)
            if not mimic_pipeline == 'teacher':
                model.save(filename + '.h5')
            with open(filename + '_history.pickle', 'wb') as f:
                # save model history
                pickle.dump(model_history, f, pickle.HIGHEST_PROTOCOL)
        #if not mimic_pipeline == 'teacher':
            #fig = plt.figure(figsize=(20,3))
            #epoch_number = range(1,len(model_history['train'])+1)
            #plt.plot(epoch_number, model_history['train'], 'g', label='Training MSE')
            #plt.plot(epoch_number, model_history['val'], 'b', label='validation MSE')
            #plt.title('Training and Validation loss')
            #plt.xlabel('Epochs')
            #plt.ylabel('MSE')
            #plt.xticks(epoch_number,epoch_number)
            #plt.title(mimic_pipeline +' '+PH+' min')
            #plt.legend()
            #plt.savefig(filename+'.png')

    #calculate Mean and STD of RMSE for different trials
    exclude_keys = list()
    for key in overall_results.keys():
        if 'RMSE' not in key:
            exclude_keys.append(key)

    rmse_df = overall_results.drop(exclude_keys, axis=1)
    
    overall_results['Mean_RMSE']= rmse_df.mean(axis=1) 
    overall_results['STD_RMSE']= rmse_df.std(axis=1) 

    #calculate Mean and STD of MAE for different trials
    exclude_keys = list()
    for key in overall_results.keys():
        if 'MAE' not in key:
            exclude_keys.append(key)
    mae_df = overall_results.drop(exclude_keys, axis=1)

    overall_results['Mean_MAE']= mae_df.mean(axis=1) 
    overall_results['STD_MAE']= mae_df.std(axis=1) 
    #total_mean = overall_results.mean(axis=0)
    #df2 = pd.DataFrame([total_mean])
    #df3 = pd.DataFrame([None]*len(overall_results.columns)).T
    #df3.columns = overall_results.columns
    #overall_results = overall_results.append(df3)
    #overall_results = overall_results.append(df2)
    
    #save overall results
    if save_results:
        filename = output_directory+mimic_pipeline+'/'+prediction_type+'_'+dimension+'_'+PH+'/'+model_name+'/%s_%s_%s_%smin' %(prediction_type,dimension,mimic_pipeline,PH)
        overall_results.to_csv(filename+'.csv')
    
    
if __name__ == "__main__":
    if len(sys.argv) > 4:
        root_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
        model_directory = sys.argv[4]
        history_window = int(sys.argv[5]) #12
        prediction_window = int(sys.argv[6]) #30 or 60 minutes
        dimension = sys.argv[7] #univariate or multivariate
        prediction_type = sys.argv[8] #single or multi (single-step or multi-output)
        if sys.argv[9] == 'False':
            normalize_data = False
        else:
            normalize_data = True
        model_name = sys.argv[10]
        dataset = sys.argv[11]
        if sys.argv[12] == 'False':
            save_results = False
        else:
            save_results = True
        mimic_pipeline = sys.argv[13] #student/teacher/retrain/teacher_student
        PH = str(prediction_window) #prediction horizon
        if prediction_window == 30 or prediction_window == 60:
            prediction_window = prediction_window//5
        
        if dimension == 'univariate':
            n_features = 1
        elif dimension == 'multivariate':
            n_features = 5
        print("Starting experiments for the following settings: prediction_window: "+PH+"; dimension: "+dimension+"; prediction_type: "+prediction_type+"; normalize_data: "+str(normalize_data)+"; model_name: "+model_name+" ;dataset: "+dataset+"; save_results: "+str(save_results)+'; mimic pipeline: '+mimic_pipeline+'\n')
        main(model_name) 
        print("Experiments completed for the following settings: prediction_window: "+PH+"; dimension: "+dimension+"; prediction_type: "+prediction_type+"; normalize_data: "+str(normalize_data)+"; model_name: "+model_name+" ;dataset: "+dataset+"; save_results: "+str(save_results)+'; mimic pipeline: '+mimic_pipeline+'\n')
    else:
        print("Invalid input arguments")
        exit(-1)
