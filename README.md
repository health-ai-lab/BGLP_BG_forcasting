# BGLP_BG_forecasting_code
Code for BGLP2020 paper on glucose prediction using OpenAPS and OhioT1DM data  

Usage:
In the terminal run the following commands twice. First with `dataset='oaps'` and then with `dataset='ohio'` in `preprocess.sh` and `run.sh` files:  

`chmod +x ./preprocess.sh`  
`./preprocess.sh`   

`chmod +x ./run.sh`  
`./run.sh`  

`chmod +x generate_bglp_results.sh`  
`./generate_bglp_results.sh`  

In shell files, set the following parameters according to the experiment you
are running.  
`dataset='ohio' (ohio or oaps)`   
`root_directory = "../../../../PHI/PHI_OHIO/" (the root folder)`   
`data_directory= $root_directory"data/" (containing raw data)`  
`output_directory="OHIO_models/" (folder to save the results in)`  
`model_directory='OAPS_models/' (folder containing model pretrained on OAPS data`  
`filter_data = 'True' (True or False - use median filter to smooth training data)`  
`normalize_data = False (True or False - normalize all feature values to be in the range of the BG levels)`    
`threshold = 15 (integer value - removes data for which BG levels are below this value)`   
`history_window = 12 (integer value - number of past glucose values to use. 12 samples means an hour of previous data (frequency = 5 minutes)`     
`prediction_window = 60 (30 or 60 minutes - prediction horizon for BG forecasting (in minutes))`  
`dimension = multivariate (univariate or multivariate)`  
`prediction_type = single (single or multi. This refers to single step or multioutput forecasting)`  
`model_name = RNN (['LSTM', 'RNN'])`  
`save_results = False (True or False. It will replace old output files in the output directory)`  
`mimic_pipeline = "student" (student, teacher, retrain, teacher_student)`  
`test_extrapolation = True (True or False - to extrapolate test data or not. Read report.docx for more info)`  
