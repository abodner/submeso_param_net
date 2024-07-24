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

    
def Jacobian_norm(x,y):
    if y.shape[1] != 1:
        print('wrong shape')
  
    dydx = torch.zeros(x.shape[1])

    grad = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        allow_unused=True, retain_graph=True, create_graph=True)[0]
      
    if grad.shape != x.shape:
        print('Error in dimensions')

    return torch.mean(grad**2, dim=[-2,-1]).mean(dim=0)



res_strg = ['1_12','1_8','1_4','1_2','1']
jacobian_ftr_res = np.empty((5,9))

for i_res in range(5):
    
    submeso_dataset=dataset.SubmesoDataset(['grad_B','FCOR', 'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'], res=res_strg[i_res])

    test_loader=DataLoader(
    submeso_dataset,
    batch_size=len(submeso_dataset.test_ind),
    sampler=submeso_dataset.test_ind)

    model = torch.load('/scratch/ab10313/pleiades/trained_models/fcnn_k5_l7_res_'+res_strg[i_res]+'.pt')

    
    for x_data, y_data in test_loader:
        x_input= x_data
        x_input.requires_grad = True
        y_output = model(x_input.to(device))
    
    jacobian_norm = Jacobian_norm(x_input,y_output)
    jacobian_ftr_res[i_res,:] = jacobian_norm.detach().numpy()

    
np.save('/scratch/ab10313/pleiades/trained_models/jacobian_ftr_res.npy',jacobian_ftr_res)
