import cartopy 
import glob
import matplotlib.pyplot as plt
import os
import xarray as xr
import xgcm 
from fastjmd95 import jmd95numba 
import numpy as np
from scipy.stats import pearsonr
import torch.utils.data as Data
from torch.utils.data import DataLoader

BASE = '/scratch/ab10313/pleiades/'

PATH_LIST_PP = glob.glob(BASE+'*/preprcossed_data/')


import submeso_ml.data.dataset as dataset


res_strg = ['1_12','1_8','1_4','1_2','1']

corr_mat_res = np.zeros((5,10,10))

for i_res in range(5):
    
    submeso_dataset=dataset.SubmesoDataset(['grad_B','FCOR',  'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'], res=res_strg[i_res])


    test_loader=DataLoader(
        submeso_dataset,
        num_workers=10,
        batch_size=len(submeso_dataset.test_ind),
        sampler=submeso_dataset.test_ind)

    for x_data, y_data in test_loader:
        x_norm = x_data.detach().numpy()
        y_norm = y_data.detach().numpy()

    xy_norm = np.concatenate([x_norm,y_norm],axis=1)
    
    corr_mat = np.zeros((10,10))
    xy_norm[:,5,:,:] = -xy_norm[:,5,:,:]
    
    for i in range(10):
        for j in range(10):
            corr_mat[i,j],_ = pearsonr(xy_norm[:,i,:,:].ravel(),xy_norm[:,j,:,:].ravel())

    corr_mat_res[i_res,:,:] = corr_mat[:,:]


np.save('/scratch/ab10313/pleiades/trained_models/corr_map.npy',corr_mat_res)
