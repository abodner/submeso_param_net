#!/usr/bin/env python
# coding: utf-8

import wandb
import numpy as np
import sys
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch.nn as nn
import glob
import xarray as xr
import matplotlib.pyplot as plt


BASE = '/scratch/ab10313/pleiades/'
save_path=BASE+"trained_models"


import submeso_ml.systems.regression_system as regression_system
import submeso_ml.models.fcnn as fcnn
import submeso_ml.data.dataset as dataset


# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

    
    
    
res_strg = ['1_12','1_8','1_4','1_2','1']

r2_cnn_monthly_mean = np.empty((5,12,12))
r2_cnn_monthly_mean[:] = np.nan
r2_param_monthly_mean = np.empty((5,12,12))
r2_param_monthly_mean[:] = np.nan

for i_res in range(5):
    
    submeso_dataset=dataset.SubmesoDataset(['grad_B','FCOR',  'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'], res=res_strg[i_res])


    test_loader=DataLoader(
        submeso_dataset,
        num_workers=10,
        batch_size=len(submeso_dataset.test_ind),
        sampler=submeso_dataset.test_ind)

    
    # load trained models
    model= torch.load('/scratch/ab10313/pleiades/trained_models/fcnn_k5_l7_res_'+res_strg[i_res]+'_tmp.pt')
    
    WB_FK08_param = 0.07*np.load(BASE+'NN_data_'+res_strg[i_res]+'/WB_FK08_param.npy')[submeso_dataset.test_ind] 

    for x_data, y_data in test_loader:
        prediction = model(x_data.to(device)).detach().numpy() 
        target = y_data.detach().numpy()
                              
    baseline_mse =  np.mean((target.flatten() - target.mean()) ** 2)                       
                              

    r2_cnn_timeseries = np.zeros(2030)
    r2_param_timeseries = np.zeros(2030)
    
    for it in range(2030):
        r2_cnn_timeseries[it] = 1 - mean_squared_error(prediction[it,:,:].flatten(),target[it,:,:].flatten())/baseline_mse
        r2_param_timeseries[it] = 1 - mean_squared_error(WB_FK08_param[it,:,:].flatten(),target[it,:,:].flatten())/baseline_mse

    
    #locations
    location_index = np.zeros(846*12)
    location_month_index= np.zeros(846*12)
    
    
    for i in range(12):
        for j in range(846):
            location_index[i*846+j] = i
            location_month_index[i*846+j] = np.floor(j/60)

 
    test_ind = submeso_dataset.test_ind
    
    r2_cnn_timeseries_loc = np.empty((12,2030))
    r2_cnn_timeseries_loc[:] = np.nan
    r2_param_timeseries_loc = np.empty((12,2030))
    r2_param_timeseries_loc[:] = np.nan
    
    location_month_index_test = location_month_index[test_ind]
    location_month_index_loc = np.empty((12,2030))
    location_month_index_loc[:] = np.nan
    
    for i_loc in range(12):    
        r2_cnn_timeseries_loc[i_loc,location_index[test_ind]==i_loc] = r2_cnn_timeseries[location_index[test_ind]==i_loc]
        r2_param_timeseries_loc[i_loc,location_index[test_ind]==i_loc] = r2_param_timeseries[location_index[test_ind]==i_loc]
        location_month_index_loc[i_loc,location_index[test_ind]==i_loc] = location_month_index_test[location_index[test_ind]==i_loc]
    


    r2_cnn_timeseries_loc_time_mean = np.empty((12,12))
    r2_param_timeseries_loc_time_mean = np.empty((12,12))
    
    for i_loc in range(12):
        for i_month in range(12):
            r2_cnn_timeseries_loc_time_mean[i_loc,i_month] = np.mean(r2_cnn_timeseries_loc[i_loc,~np.isnan(r2_cnn_timeseries_loc[i_loc])][location_month_index_loc[i_loc,~np.isnan(location_month_index_loc[i_loc])]==i_month])
            
            r2_param_timeseries_loc_time_mean[i_loc,i_month] = np.mean(r2_param_timeseries_loc[i_loc,~np.isnan(r2_param_timeseries_loc[i_loc])][location_month_index_loc[i_loc,~np.isnan(location_month_index_loc[i_loc])]==i_month])


    r2_cnn_monthly_mean[i_res,:,:] = r2_cnn_timeseries_loc_time_mean[:,:]
    r2_param_monthly_mean[i_res,:,:] = r2_param_timeseries_loc_time_mean[:,:]

np.save('/scratch/ab10313/pleiades/trained_models/r2_cnn_monthly_mean.npy',r2_cnn_monthly_mean)
np.save('/scratch/ab10313/pleiades/trained_models/r2_param_monthly_mean.npy',r2_param_monthly_mean)
