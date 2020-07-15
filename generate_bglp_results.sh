#!/bin/sh
test_extrapolation='True'
root_directory="../../../../PHI/PHI_OHIO/" 
data_directory=$root_directory"data/" 
if [ "$test_extrapolation" = "True" ]; then
    output_directory='bglp2020_results/with_extrapolation/'
fi
if [ "$test_extrapolation" = "False" ]; then
    output_directory='bglp2020_results/without_extrapolation/'
fi
model_directory='OHIO_models/'
pipeline='student'
prediction_window=30
model_name='RNN'
history_window=12 
prediction_type=single #single-step or multi-output
normalize_data=False
model_name=LSTM
save_results=False
mimic_pipeline="student"
if [ "$test_extrapolation" = "True" ]; then
    python $PWD/with_extrapolation.py $data_directory $output_directory $pipeline $prediction_window $model_name $history_window $model_directory
fi
if [ "$test_extrapolation" = "False" ]; then
    python $PWD/without_extrapolation.py $data_directory $output_directory $pipeline $dimension $prediction_window $model_name $history_window $model_directory
fi