Files
******************************************************
report.pdf:
summary of the results and the methods used
******************************************************
<system_ID>_<dc_ID>_<horizon>.txt (e.g.  I_540_30.text for 30 PH = minutes and I_540_60.text for 60 PH = minutes for subject with ID 540): 
Contains datestamps and estimated values. We tried 4 learning pipeline. The system ID represents the pipeline that gave the best results, in this case pipeline I.
******************************************************
error_grid_plot_<PH>.png: Clarke’s error grid for each prediction horizon
******************************************************
results_summary_<PH>.csv: summary of results for each prediction horizon. Reports RMSE and AME. The experiments were repeated for 10 times and the average RMSE and MAE was tabulated for each subject.
******************************************************
results_dictionary_<PH>.pickle: A dictionary that contains average RMSE, MAE and a list of predicted BG values for each of the six subjects. 
******************************************************
 
