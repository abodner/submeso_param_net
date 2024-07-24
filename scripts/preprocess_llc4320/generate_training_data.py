#!/usr/bin/env python
# coding: utf-8

# # Generate NN data at filter scales of: 1/12, 1/8, 1/4, 1/2, 1
# We stack all regions in one array and save global normalization factors (global mean and std) to be used as part of the dataloader 

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import glob
from xgcm import Grid


#path
BASE = '/scratch/ab10313/pleiades/'
PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/')


# coarse grain data
def coarse_grain(data, grid_factor):
    if len(data.dims) == 3:
        data_cg = data.coarsen(i=grid_factor,j=grid_factor, boundary="trim").mean()
    elif len(data.dims) == 2:
        data_cg = np.tile(data.coarsen(i=grid_factor,j=grid_factor, boundary="trim").mean(),(859,1,1))
    return data_cg


# normalization factors
def global_mean_std(data):
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    return data_mean, data_std


# splits datasets along the spatial axes and concats them back into single array under time
def load_data(var_name_string,grid_factor):
    PATH_LIST = glob.glob(BASE+'/preprocessed_data/'+var_name_string+'.nc') 
    data_0 = xr.open_dataarray(PATH_LIST[0])
    data_smooth_0 = coarse_grain(data_0,grid_factor)
    data_app = data_smooth_0
    for i_file in np.arange(1,len(PATH_LIST)):
        PATH = PATH_LIST[i_file]
        data = xr.open_dataarray(PATH)
        data_smooth = coarse_grain(data,grid_factor)
        data_app = np.concatenate((data_app,data_smooth),axis=0)
    return data_app




def WB_sg_target(PATH,grid_factor):
    # WB
    B = coarse_grain(xr.open_dataarray(PATH+'B.nc'),grid_factor).values
    W = coarse_grain(xr.open_dataarray(PATH+'W.nc'),grid_factor).values
    WB = coarse_grain(xr.open_dataarray(PATH+'WB.nc'),grid_factor).values
    
    # WB subgrid
    WB_sg = WB - W*B
    return WB_sg


    
def load_data_WB(grid_factor):
    PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/') 
    WB_sg_0 = WB_sg_target(PATH_LIST_full[0],grid_factor)
    data_app = WB_sg_0
    for i_file in np.arange(1,len(PATH_LIST_full)):
        WB_sg = WB_sg_target(PATH_LIST_full[i_file],grid_factor)
        data_app = np.concatenate((data_app,WB_sg),axis=0)
    return data_app




def grad_B_mag(PATH,grid_factor):
    # grad_B
    B = coarse_grain(xr.open_dataarray(PATH+'B.nc'),grid_factor=grid_factor)
    B_x = (B.diff(dim='i')/(grid_factor*B.dxF)).interp(i=B.i,j=B.j,kwargs={"fill_value": "extrapolate"})
    B_y = (B.diff(dim='j')/(grid_factor*B.dyF)).interp(i=B.i,j=B.j,kwargs={"fill_value": "extrapolate"})
    grad_B = np.sqrt(B_y**2 + B_x**2).values
    return grad_B

    
def load_data_grad_B(grid_factor):
    PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/') 
    grad_B_0 = grad_B_mag(PATH_LIST_full[0],grid_factor)
    data_app = grad_B_0
    for i_file in np.arange(1,len(PATH_LIST_full)):
        grad_B = grad_B_mag(PATH_LIST_full[i_file],grid_factor)
        data_app = np.concatenate((data_app,grad_B),axis=0)
    return data_app



def vel_gradients(grid_factor):
    PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/') 
    # coarse-res gradients
    U = coarse_grain(xr.open_dataarray(PATH_LIST_full[0]+'U.nc'),grid_factor=grid_factor)
    U_x = (U.diff(dim='i')/(grid_factor*U.dxF)).interp(i=U.i,j=U.j,kwargs={"fill_value": "extrapolate"}).values
    U_y = (U.diff(dim='j')/(grid_factor*U.dyF)).interp(i=U.i,j=U.j,kwargs={"fill_value": "extrapolate"}).values
    
    V = coarse_grain(xr.open_dataarray(PATH_LIST_full[0]+'V.nc'),grid_factor=grid_factor)
    V_x = (V.diff(dim='i')/(grid_factor*V.dxF)).interp(i=V.i,j=V.j,kwargs={"fill_value": "extrapolate"}).values
    V_y = (V.diff(dim='j')/(grid_factor*V.dyF)).interp(i=V.i,j=V.j,kwargs={"fill_value": "extrapolate"}).values

    
    # divergence
    div_0 = U_x + V_y
    div_app = div_0
    
    # vorticity
    vort_0= V_x - U_y
    vort_app = vort_0
    
    # strain
    strain_0 = np.sqrt((U_x - V_y)**2 + (V_x + U_y)**2)
    strain_app = strain_0

    for i_file in np.arange(1,len(PATH_LIST_full)):
        # coarse-res gradients
        U = coarse_grain(xr.open_dataarray(PATH_LIST_full[i_file]+'U.nc'),grid_factor=grid_factor)
        U_x = (U.diff(dim='i')/(grid_factor*U.dxF)).interp(i=U.i,j=U.j,kwargs={"fill_value": "extrapolate"}).values
        U_y = (U.diff(dim='j')/(grid_factor*U.dyF)).interp(i=U.i,j=U.j,kwargs={"fill_value": "extrapolate"}).values

        V = coarse_grain(xr.open_dataarray(PATH_LIST_full[i_file]+'V.nc'),grid_factor=grid_factor)
        V_x = (V.diff(dim='i')/(grid_factor*V.dxF)).interp(i=V.i,j=V.j,kwargs={"fill_value": "extrapolate"}).values
        V_y = (V.diff(dim='j')/(grid_factor*V.dyF)).interp(i=V.i,j=V.j,kwargs={"fill_value": "extrapolate"}).values

        # divergence
        div = U_x + V_y
        div_app = np.concatenate((div_app,div),axis=0)

        # vorticity
        vort = V_x - U_y
        vort_app = np.concatenate((vort_app,vort),axis=0)

        # strain
        strain = np.sqrt((U_x - V_y)**2 + (V_x + U_y)**2)
        strain_app = np.concatenate((strain_app,strain),axis=0)
    
                             
    return div_app, vort_app, strain_app



def TAU_mag(PATH,grid_factor):
    # wind stress
    TAUX = coarse_grain(xr.open_dataarray(PATH+'TAUX.nc'),grid_factor=grid_factor)
    TAUY = coarse_grain(xr.open_dataarray(PATH+'TAUY.nc'),grid_factor=grid_factor)
    TAU = np.sqrt(TAUY**2 + TAUX**2).values

    return TAU


    
def load_data_TAU_mag(grid_factor):
    PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/') 
    TAU_0 = TAU_mag(PATH_LIST_full[0],grid_factor)
    data_app = TAU_0
    for i_file in np.arange(1,len(PATH_LIST_full)):
        TAU = TAU_mag(PATH_LIST_full[i_file],grid_factor)
        data_app = np.concatenate((data_app,TAU),axis=0)
    return data_app



def load_data_drF(grid_factor):
    # cell size for weighted average
    PATH_LIST = glob.glob(BASE+'/preprocessed_data/B.nc') 
    B_0 = xr.open_dataarray(PATH_LIST[0])
    data_0 = (B_0.dxF**2 + B_0.dyF**2)**0.5
    data_smooth_0 = coarse_grain(data_0,grid_factor=grid_factor)
    data_app = data_smooth_0
    for i_file in np.arange(1,len(PATH_LIST)):
        PATH = PATH_LIST[i_file]
        B_i = xr.open_dataarray(PATH)
        data = (B_i.dxF**2 + B_i.dyF**2)**0.5
        data_smooth = coarse_grain(data,grid_factor=grid_factor)
        data_app = np.concatenate((data_app,data_smooth),axis=0)
    return data_app



def frontal_width_23(PATH,grid_factor):
    # coefficients from Bodner 2023 
    Q = coarse_grain(xr.open_dataarray(PATH+'Q.nc'),grid_factor).values
    HBL = coarse_grain(xr.open_dataarray(PATH+'HBL.nc'),grid_factor).values
    FCOR = coarse_grain(xr.open_dataarray(PATH+'FCOR.nc'),grid_factor)
    TAUX = coarse_grain(xr.open_dataarray(PATH+'TAUX.nc'),grid_factor=grid_factor)
    TAUY = coarse_grain(xr.open_dataarray(PATH+'TAUY.nc'),grid_factor=grid_factor)
    TAU = np.sqrt(TAUY**2 + TAUX**2).values
    
    m_star = 0.5
    n_star = 0.066
    Cl = 0.25
    rho0 = 1000
    # u_star
    u_star = np.sqrt(TAU/rho0)
    # w_star
    cp_w = 4.2e3 
    galpha = 1.962e-3              
    Q_neg = np.where(Q < 0, Q, 0)
    B0 = (galpha/rho0/cp_w )*Q_neg
    w_star = (-B0*HBL)**(1/3) 
    
    Lf = (Cl *(m_star*u_star**3 + n_star*w_star**3)**(2/3))/((FCOR**2)*(HBL))
    return Lf

def frontal_width_11(PATH,grid_factor):
    tau = 86400 
    HML = coarse_grain(xr.open_dataarray(PATH+'HML.nc'),grid_factor).values
    FCOR = coarse_grain(xr.open_dataarray(PATH+'FCOR.nc'),grid_factor)
    Nsquared = coarse_grain(xr.open_dataarray(PATH+'Nsquared.nc'),grid_factor).values

    Lf = (np.sqrt(Nsquared)*HML)/np.sqrt(FCOR**2 + tau**-2)
    return Lf


def load_data_Lf(grid_factor):
    PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/') 
    Lf_23_0 = frontal_width_23(PATH_LIST_full[0],grid_factor)
    data_app_23 = Lf_23_0
    Lf_11_0 = frontal_width_11(PATH_LIST_full[0],grid_factor)
    data_app_11 = Lf_11_0
    for i_file in np.arange(1,len(PATH_LIST_full)):
        Lf_23 = frontal_width_23(PATH_LIST_full[i_file],grid_factor)
        data_app_23 = np.concatenate((data_app_23,Lf_23),axis=0)
        Lf_11 = frontal_width_11(PATH_LIST_full[i_file],grid_factor)
        data_app_11 = np.concatenate((data_app_11,Lf_11),axis=0)

    return data_app_23, data_app_11


def WB_param(PATH,grid_factor):
    # WB
    B = coarse_grain(xr.open_dataarray(PATH+'B.nc'),grid_factor=grid_factor)
    B_x = (B.diff(dim='i')/(grid_factor*B.dxF)).interp(i=B.i,j=B.j,kwargs={"fill_value": "extrapolate"}).values
    B_y = (B.diff(dim='j')/(grid_factor*B.dyF)).interp(i=B.i,j=B.j,kwargs={"fill_value": "extrapolate"}).values
    HML = coarse_grain(xr.open_dataarray(PATH+'HML.nc'),grid_factor).values
    FCOR = coarse_grain(xr.open_dataarray(PATH+'FCOR.nc'),grid_factor)
    tau = 86400 
    
    Psi_FK08_param_x = ((HML**2) * B_y)/np.sqrt(FCOR**2+ tau**-2)
    Psi_FK08_param_y = -((HML**2) * B_x)/np.sqrt(FCOR**2+ tau**-2)
    WB_FK11_param = (Psi_FK08_param_x*B_y - Psi_FK08_param_y*B_x)
    
    return WB_FK11_param


    
def load_data_WB_param(grid_factor):
    PATH_LIST_full = glob.glob(BASE+'/preprocessed_data/') 
    WB_FK_0 = WB_param(PATH_LIST_full[0],grid_factor)
    data_app = WB_FK_0
    for i_file in np.arange(1,len(PATH_LIST_full)):
        WB_FK = WB_param(PATH_LIST_full[i_file],grid_factor)
        data_app = np.concatenate((data_app,WB_FK),axis=0)

    return data_app




# ## Save training data

# ## 1/12 degree coarse-graining
grid_factor = 4

 
# coarse-res horizontal buoyancy gradient 
grad_B = load_data_grad_B(grid_factor=grid_factor)
grad_B_mean, grad_B_std = global_mean_std(grad_B)

# Coarse-res strain, vorticity, divergence
div, vort, strain = vel_gradients(grid_factor=grid_factor)
div_mean, div_std = global_mean_std(div)
vort_mean, vort_std = global_mean_std(vort)
strain_mean, strain_std = global_mean_std(strain)

# WB
WB_sg = load_data_WB(grid_factor=grid_factor)
WB_sg_mean, WB_sg_std = global_mean_std(WB_sg)

# Coriolis
FCOR = load_data('FCOR',grid_factor=grid_factor) 
FCOR_mean, FCOR_std = global_mean_std(FCOR)

# H mixed layer (NEGATIVE)
HML = load_data('HML',grid_factor=grid_factor) 
HML_mean, HML_std = global_mean_std(HML)

# Nsquared 
Nsquared = load_data('Nsquared',grid_factor=grid_factor)
Nsquared_mean, Nsquared_std = global_mean_std(Nsquared)

# wind stress
TAU   = load_data_TAU_mag(grid_factor=grid_factor)
TAU_mean, TAU_std = global_mean_std(TAU)

# surface heat flux
Q = load_data('Q',grid_factor=grid_factor)
Q_mean, Q_std = global_mean_std(Q)

# H boundary layer (POSITIVE)
HBL = load_data('HBL',grid_factor=grid_factor)
HBL_mean, HBL_std = global_mean_std(HBL)

# cell size for weighted average
drF = load_data_drF(grid_factor=grid_factor)
 
# Fox-Kemper 2008 parmaeterization 
WB_FK08_param = load_data_WB_param(grid_factor=grid_factor)

# Fox-Kemper 2011 and Bodner 2023 frontal width rescaling factors
Lf_Bodner_23, Lf_FK_11 = load_data_Lf(grid_factor=grid_factor)


# save normalized NN data
SAVE_PATH = BASE+'NN_data_1_12/'
os.mkdir(SAVE_PATH)


np.save(SAVE_PATH+'grad_B.npy',grad_B)
np.save(SAVE_PATH+'grad_B_mean.npy',grad_B_mean)
np.save(SAVE_PATH+'grad_B_std.npy',grad_B_std)

np.save(SAVE_PATH+'FCOR.npy',FCOR)
np.save(SAVE_PATH+'FCOR_mean.npy',FCOR_mean)
np.save(SAVE_PATH+'FCOR_std.npy',FCOR_std)

np.save(SAVE_PATH+'TAU.npy',TAU)
np.save(SAVE_PATH+'TAU_mean.npy',TAU_mean)
np.save(SAVE_PATH+'TAU_std.npy',TAU_std)

np.save(SAVE_PATH+'Q.npy',Q)
np.save(SAVE_PATH+'Q_mean.npy',Q_mean)
np.save(SAVE_PATH+'Q_std.npy',Q_std)

np.save(SAVE_PATH+'Nsquared.npy',Nsquared)
np.save(SAVE_PATH+'Nsquared_mean.npy',Nsquared_mean)
np.save(SAVE_PATH+'Nsquared_std.npy',Nsquared_std)

np.save(SAVE_PATH+'HML.npy',HML)
np.save(SAVE_PATH+'HML_mean.npy',HML_mean)
np.save(SAVE_PATH+'HML_std.npy',HML_std)

np.save(SAVE_PATH+'HBL.npy',HBL)
np.save(SAVE_PATH+'HBL_mean.npy',HBL_mean)
np.save(SAVE_PATH+'HBL_std.npy',HBL_std)

np.save(SAVE_PATH+'strain.npy',strain)
np.save(SAVE_PATH+'strain_mean.npy',strain_mean)
np.save(SAVE_PATH+'strain_std.npy',strain_std)

np.save(SAVE_PATH+'div.npy',div)
np.save(SAVE_PATH+'div_mean.npy',div_mean)
np.save(SAVE_PATH+'div_std.npy',div_std)

np.save(SAVE_PATH+'vort.npy',vort)
np.save(SAVE_PATH+'vort_mean.npy',vort_mean)
np.save(SAVE_PATH+'vort_std.npy',vort_std)

np.save(SAVE_PATH+'WB_sg.npy',WB_sg)
np.save(SAVE_PATH+'WB_sg_mean.npy',WB_sg_mean)
np.save(SAVE_PATH+'WB_sg_std.npy',WB_sg_std)
 
np.save(SAVE_PATH+'WB_FK08_param.npy',WB_FK08_param)
np.save(SAVE_PATH+'Lf_Bodner_23.npy',Lf_Bodner_23)
np.save(SAVE_PATH+'Lf_FK_11.npy',Lf_FK_11)
 
# grid cell size for weighted average
np.save(SAVE_PATH+'drF.npy',drF)
 



# ## 1/8 degree coarse-graining
grid_factor = 6

 
# coarse-res horizontal buoyancy gradient 
grad_B = load_data_grad_B(grid_factor=grid_factor)
grad_B_mean, grad_B_std = global_mean_std(grad_B)

# Coarse-res strain, vorticity, divergence
div, vort, strain = vel_gradients(grid_factor=grid_factor)
div_mean, div_std = global_mean_std(div)
vort_mean, vort_std = global_mean_std(vort)
strain_mean, strain_std = global_mean_std(strain)

# WB
WB_sg = load_data_WB(grid_factor=grid_factor)
WB_sg_mean, WB_sg_std = global_mean_std(WB_sg)

# Coriolis
FCOR = load_data('FCOR',grid_factor=grid_factor) 
FCOR_mean, FCOR_std = global_mean_std(FCOR)

# H mixed layer (NEGATIVE)
HML = load_data('HML',grid_factor=grid_factor) 
HML_mean, HML_std = global_mean_std(HML)

# Nsquared 
Nsquared = load_data('Nsquared',grid_factor=grid_factor)
Nsquared_mean, Nsquared_std = global_mean_std(Nsquared)

# wind stress
TAU   = load_data_TAU_mag(grid_factor=grid_factor)
TAU_mean, TAU_std = global_mean_std(TAU)

# surface heat flux
Q = load_data('Q',grid_factor=grid_factor)
Q_mean, Q_std = global_mean_std(Q)

# H boundary layer (POSITIVE)
HBL = load_data('HBL',grid_factor=grid_factor)
HBL_mean, HBL_std = global_mean_std(HBL)

# cell size for weighted average
drF = load_data_drF(grid_factor=grid_factor)
 
# Fox-Kemper 2008 parmaeterization 
WB_FK08_param = load_data_WB_param(grid_factor=grid_factor)

# Fox-Kemper 2011 and Bodner 2023 frontal width rescaling factors
Lf_Bodner_23, Lf_FK_11 = load_data_Lf(grid_factor=grid_factor)

# save normalized NN data
SAVE_PATH = BASE+'NN_data_1_8/'
os.mkdir(SAVE_PATH)
 

np.save(SAVE_PATH+'grad_B.npy',grad_B)
np.save(SAVE_PATH+'grad_B_mean.npy',grad_B_mean)
np.save(SAVE_PATH+'grad_B_std.npy',grad_B_std)

np.save(SAVE_PATH+'FCOR.npy',FCOR)
np.save(SAVE_PATH+'FCOR_mean.npy',FCOR_mean)
np.save(SAVE_PATH+'FCOR_std.npy',FCOR_std)

np.save(SAVE_PATH+'TAU.npy',TAU)
np.save(SAVE_PATH+'TAU_mean.npy',TAU_mean)
np.save(SAVE_PATH+'TAU_std.npy',TAU_std)

np.save(SAVE_PATH+'Q.npy',Q)
np.save(SAVE_PATH+'Q_mean.npy',Q_mean)
np.save(SAVE_PATH+'Q_std.npy',Q_std)

np.save(SAVE_PATH+'Nsquared.npy',Nsquared)
np.save(SAVE_PATH+'Nsquared_mean.npy',Nsquared_mean)
np.save(SAVE_PATH+'Nsquared_std.npy',Nsquared_std)

np.save(SAVE_PATH+'HML.npy',HML)
np.save(SAVE_PATH+'HML_mean.npy',HML_mean)
np.save(SAVE_PATH+'HML_std.npy',HML_std)

np.save(SAVE_PATH+'HBL.npy',HBL)
np.save(SAVE_PATH+'HBL_mean.npy',HBL_mean)
np.save(SAVE_PATH+'HBL_std.npy',HBL_std)

np.save(SAVE_PATH+'strain.npy',strain)
np.save(SAVE_PATH+'strain_mean.npy',strain_mean)
np.save(SAVE_PATH+'strain_std.npy',strain_std)

np.save(SAVE_PATH+'div.npy',div)
np.save(SAVE_PATH+'div_mean.npy',div_mean)
np.save(SAVE_PATH+'div_std.npy',div_std)

np.save(SAVE_PATH+'vort.npy',vort)
np.save(SAVE_PATH+'vort_mean.npy',vort_mean)
np.save(SAVE_PATH+'vort_std.npy',vort_std)

np.save(SAVE_PATH+'WB_sg.npy',WB_sg)
np.save(SAVE_PATH+'WB_sg_mean.npy',WB_sg_mean)
np.save(SAVE_PATH+'WB_sg_std.npy',WB_sg_std)
 
np.save(SAVE_PATH+'WB_FK08_param.npy',WB_FK08_param)
np.save(SAVE_PATH+'Lf_Bodner_23.npy',Lf_Bodner_23)
np.save(SAVE_PATH+'Lf_FK_11.npy',Lf_FK_11)
 
# grid cell size for weighted average
np.save(SAVE_PATH+'drF.npy',drF)
 







# ## 1/4 degree coarse-graining
grid_factor = 12

# coarse-res horizontal buoyancy gradient 
grad_B = load_data_grad_B(grid_factor=grid_factor)
grad_B_mean, grad_B_std = global_mean_std(grad_B)

# Coarse-res strain, vorticity, divergence
div, vort, strain = vel_gradients(grid_factor=grid_factor)
div_mean, div_std = global_mean_std(div)
vort_mean, vort_std = global_mean_std(vort)
strain_mean, strain_std = global_mean_std(strain)

# WB
WB_sg = load_data_WB(grid_factor=grid_factor)
WB_sg_mean, WB_sg_std = global_mean_std(WB_sg)

# Coriolis
FCOR = load_data('FCOR',grid_factor=grid_factor) 
FCOR_mean, FCOR_std = global_mean_std(FCOR)

# H mixed layer (NEGATIVE)
HML = load_data('HML',grid_factor=grid_factor) 
HML_mean, HML_std = global_mean_std(HML)

# Nsquared 
Nsquared = load_data('Nsquared',grid_factor=grid_factor)
Nsquared_mean, Nsquared_std = global_mean_std(Nsquared)

# wind stress
TAU   = load_data_TAU_mag(grid_factor=grid_factor)
TAU_mean, TAU_std = global_mean_std(TAU)

# surface heat flux
Q = load_data('Q',grid_factor=grid_factor)
Q_mean, Q_std = global_mean_std(Q)

# H boundary layer (POSITIVE)
HBL = load_data('HBL',grid_factor=grid_factor)
HBL_mean, HBL_std = global_mean_std(HBL)

# cell size for weighted average
drF = load_data_drF(grid_factor=grid_factor)
 
# Fox-Kemper 2008 parmaeterization 
WB_FK08_param = load_data_WB_param(grid_factor=grid_factor)

# Fox-Kemper 2011 and Bodner 2023 frontal width rescaling factors
Lf_Bodner_23, Lf_FK_11 = load_data_Lf(grid_factor=grid_factor)


# save normalized NN data
SAVE_PATH = BASE+'NN_data_1_4/'
os.mkdir(SAVE_PATH)
 

np.save(SAVE_PATH+'grad_B.npy',grad_B)
np.save(SAVE_PATH+'grad_B_mean.npy',grad_B_mean)
np.save(SAVE_PATH+'grad_B_std.npy',grad_B_std)

np.save(SAVE_PATH+'FCOR.npy',FCOR)
np.save(SAVE_PATH+'FCOR_mean.npy',FCOR_mean)
np.save(SAVE_PATH+'FCOR_std.npy',FCOR_std)

np.save(SAVE_PATH+'TAU.npy',TAU)
np.save(SAVE_PATH+'TAU_mean.npy',TAU_mean)
np.save(SAVE_PATH+'TAU_std.npy',TAU_std)

np.save(SAVE_PATH+'Q.npy',Q)
np.save(SAVE_PATH+'Q_mean.npy',Q_mean)
np.save(SAVE_PATH+'Q_std.npy',Q_std)

np.save(SAVE_PATH+'Nsquared.npy',Nsquared)
np.save(SAVE_PATH+'Nsquared_mean.npy',Nsquared_mean)
np.save(SAVE_PATH+'Nsquared_std.npy',Nsquared_std)

np.save(SAVE_PATH+'HML.npy',HML)
np.save(SAVE_PATH+'HML_mean.npy',HML_mean)
np.save(SAVE_PATH+'HML_std.npy',HML_std)

np.save(SAVE_PATH+'HBL.npy',HBL)
np.save(SAVE_PATH+'HBL_mean.npy',HBL_mean)
np.save(SAVE_PATH+'HBL_std.npy',HBL_std)

np.save(SAVE_PATH+'strain.npy',strain)
np.save(SAVE_PATH+'strain_mean.npy',strain_mean)
np.save(SAVE_PATH+'strain_std.npy',strain_std)

np.save(SAVE_PATH+'div.npy',div)
np.save(SAVE_PATH+'div_mean.npy',div_mean)
np.save(SAVE_PATH+'div_std.npy',div_std)

np.save(SAVE_PATH+'vort.npy',vort)
np.save(SAVE_PATH+'vort_mean.npy',vort_mean)
np.save(SAVE_PATH+'vort_std.npy',vort_std)

np.save(SAVE_PATH+'WB_sg.npy',WB_sg)
np.save(SAVE_PATH+'WB_sg_mean.npy',WB_sg_mean)
np.save(SAVE_PATH+'WB_sg_std.npy',WB_sg_std)
 
np.save(SAVE_PATH+'WB_FK08_param.npy',WB_FK08_param)
np.save(SAVE_PATH+'Lf_Bodner_23.npy',Lf_Bodner_23)
np.save(SAVE_PATH+'Lf_FK_11.npy',Lf_FK_11)
 
# grid cell size for weighted average
np.save(SAVE_PATH+'drF.npy',drF)
 



 ## 1/2 degree coarse-graining
grid_factor = 24

 
# coarse-res horizontal buoyancy gradient 
grad_B = load_data_grad_B(grid_factor=grid_factor)
grad_B_mean, grad_B_std = global_mean_std(grad_B)

# Coarse-res strain, vorticity, divergence
div, vort, strain = vel_gradients(grid_factor=grid_factor)
div_mean, div_std = global_mean_std(div)
vort_mean, vort_std = global_mean_std(vort)
strain_mean, strain_std = global_mean_std(strain)

# WB
WB_sg = load_data_WB(grid_factor=grid_factor)
WB_sg_mean, WB_sg_std = global_mean_std(WB_sg)

# Coriolis
FCOR = load_data('FCOR',grid_factor=grid_factor) 
FCOR_mean, FCOR_std = global_mean_std(FCOR)

# H mixed layer (NEGATIVE)
HML = load_data('HML',grid_factor=grid_factor) 
HML_mean, HML_std = global_mean_std(HML)

# Nsquared 
Nsquared = load_data('Nsquared',grid_factor=grid_factor)
Nsquared_mean, Nsquared_std = global_mean_std(Nsquared)

# wind stress
TAU   = load_data_TAU_mag(grid_factor=grid_factor)
TAU_mean, TAU_std = global_mean_std(TAU)

# surface heat flux
Q = load_data('Q',grid_factor=grid_factor)
Q_mean, Q_std = global_mean_std(Q)

# H boundary layer (POSITIVE)
HBL = load_data('HBL',grid_factor=grid_factor)
HBL_mean, HBL_std = global_mean_std(HBL)

# cell size for weighted average
drF = load_data_drF(grid_factor=grid_factor)
 
# Fox-Kemper 2008 parmaeterization 
WB_FK08_param = load_data_WB_param(grid_factor=grid_factor)

# Fox-Kemper 2011 and Bodner 2023 frontal width rescaling factors
Lf_Bodner_23, Lf_FK_11 = load_data_Lf(grid_factor=grid_factor)


# save normalized NN data
SAVE_PATH = BASE+'NN_data_1_2/'
os.mkdir(SAVE_PATH)
 

np.save(SAVE_PATH+'grad_B.npy',grad_B)
np.save(SAVE_PATH+'grad_B_mean.npy',grad_B_mean)
np.save(SAVE_PATH+'grad_B_std.npy',grad_B_std)

np.save(SAVE_PATH+'FCOR.npy',FCOR)
np.save(SAVE_PATH+'FCOR_mean.npy',FCOR_mean)
np.save(SAVE_PATH+'FCOR_std.npy',FCOR_std)

np.save(SAVE_PATH+'TAU.npy',TAU)
np.save(SAVE_PATH+'TAU_mean.npy',TAU_mean)
np.save(SAVE_PATH+'TAU_std.npy',TAU_std)

np.save(SAVE_PATH+'Q.npy',Q)
np.save(SAVE_PATH+'Q_mean.npy',Q_mean)
np.save(SAVE_PATH+'Q_std.npy',Q_std)

np.save(SAVE_PATH+'Nsquared.npy',Nsquared)
np.save(SAVE_PATH+'Nsquared_mean.npy',Nsquared_mean)
np.save(SAVE_PATH+'Nsquared_std.npy',Nsquared_std)

np.save(SAVE_PATH+'HML.npy',HML)
np.save(SAVE_PATH+'HML_mean.npy',HML_mean)
np.save(SAVE_PATH+'HML_std.npy',HML_std)

np.save(SAVE_PATH+'HBL.npy',HBL)
np.save(SAVE_PATH+'HBL_mean.npy',HBL_mean)
np.save(SAVE_PATH+'HBL_std.npy',HBL_std)

np.save(SAVE_PATH+'strain.npy',strain)
np.save(SAVE_PATH+'strain_mean.npy',strain_mean)
np.save(SAVE_PATH+'strain_std.npy',strain_std)

np.save(SAVE_PATH+'div.npy',div)
np.save(SAVE_PATH+'div_mean.npy',div_mean)
np.save(SAVE_PATH+'div_std.npy',div_std)

np.save(SAVE_PATH+'vort.npy',vort)
np.save(SAVE_PATH+'vort_mean.npy',vort_mean)
np.save(SAVE_PATH+'vort_std.npy',vort_std)

np.save(SAVE_PATH+'WB_sg.npy',WB_sg)
np.save(SAVE_PATH+'WB_sg_mean.npy',WB_sg_mean)
np.save(SAVE_PATH+'WB_sg_std.npy',WB_sg_std)
 
np.save(SAVE_PATH+'WB_FK08_param.npy',WB_FK08_param)
np.save(SAVE_PATH+'Lf_Bodner_23.npy',Lf_Bodner_23)
np.save(SAVE_PATH+'Lf_FK_11.npy',Lf_FK_11)
 
# grid cell size for weighted average
np.save(SAVE_PATH+'drF.npy',drF)
 


# ## 1 degree coarse-graining
grid_factor = 48

# coarse-res horizontal buoyancy gradient 
grad_B = load_data_grad_B(grid_factor=grid_factor)
grad_B_mean, grad_B_std = global_mean_std(grad_B)

# Coarse-res strain, vorticity, divergence
div, vort, strain = vel_gradients(grid_factor=grid_factor)
div_mean, div_std = global_mean_std(div)
vort_mean, vort_std = global_mean_std(vort)
strain_mean, strain_std = global_mean_std(strain)

# WB
WB_sg = load_data_WB(grid_factor=grid_factor)
WB_sg_mean, WB_sg_std = global_mean_std(WB_sg)

# Coriolis
FCOR = load_data('FCOR',grid_factor=grid_factor) 
FCOR_mean, FCOR_std = global_mean_std(FCOR)

# H mixed layer (NEGATIVE)
HML = load_data('HML',grid_factor=grid_factor) 
HML_mean, HML_std = global_mean_std(HML)

# Nsquared 
Nsquared = load_data('Nsquared',grid_factor=grid_factor)
Nsquared_mean, Nsquared_std = global_mean_std(Nsquared)

# wind stress
TAU   = load_data_TAU_mag(grid_factor=grid_factor)
TAU_mean, TAU_std = global_mean_std(TAU)

# surface heat flux
Q = load_data('Q',grid_factor=grid_factor)
Q_mean, Q_std = global_mean_std(Q)

# H boundary layer (POSITIVE)
HBL = load_data('HBL',grid_factor=grid_factor)
HBL_mean, HBL_std = global_mean_std(HBL)

# cell size for weighted average
drF = load_data_drF(grid_factor=grid_factor)
 
# Fox-Kemper 2008 parmaeterization 
WB_FK08_param = load_data_WB_param(grid_factor=grid_factor)

# Fox-Kemper 2011 and Bodner 2023 frontal width rescaling factors
Lf_Bodner_23, Lf_FK_11 = load_data_Lf(grid_factor=grid_factor)


# save normalized NN data
SAVE_PATH = BASE+'NN_data_1/'
os.mkdir(SAVE_PATH)
 

np.save(SAVE_PATH+'grad_B.npy',grad_B)
np.save(SAVE_PATH+'grad_B_mean.npy',grad_B_mean)
np.save(SAVE_PATH+'grad_B_std.npy',grad_B_std)

np.save(SAVE_PATH+'FCOR.npy',FCOR)
np.save(SAVE_PATH+'FCOR_mean.npy',FCOR_mean)
np.save(SAVE_PATH+'FCOR_std.npy',FCOR_std)

np.save(SAVE_PATH+'TAU.npy',TAU)
np.save(SAVE_PATH+'TAU_mean.npy',TAU_mean)
np.save(SAVE_PATH+'TAU_std.npy',TAU_std)

np.save(SAVE_PATH+'Q.npy',Q)
np.save(SAVE_PATH+'Q_mean.npy',Q_mean)
np.save(SAVE_PATH+'Q_std.npy',Q_std)

np.save(SAVE_PATH+'Nsquared.npy',Nsquared)
np.save(SAVE_PATH+'Nsquared_mean.npy',Nsquared_mean)
np.save(SAVE_PATH+'Nsquared_std.npy',Nsquared_std)

np.save(SAVE_PATH+'HML.npy',HML)
np.save(SAVE_PATH+'HML_mean.npy',HML_mean)
np.save(SAVE_PATH+'HML_std.npy',HML_std)

np.save(SAVE_PATH+'HBL.npy',HBL)
np.save(SAVE_PATH+'HBL_mean.npy',HBL_mean)
np.save(SAVE_PATH+'HBL_std.npy',HBL_std)

np.save(SAVE_PATH+'strain.npy',strain)
np.save(SAVE_PATH+'strain_mean.npy',strain_mean)
np.save(SAVE_PATH+'strain_std.npy',strain_std)

np.save(SAVE_PATH+'div.npy',div)
np.save(SAVE_PATH+'div_mean.npy',div_mean)
np.save(SAVE_PATH+'div_std.npy',div_std)

np.save(SAVE_PATH+'vort.npy',vort)
np.save(SAVE_PATH+'vort_mean.npy',vort_mean)
np.save(SAVE_PATH+'vort_std.npy',vort_std)

np.save(SAVE_PATH+'WB_sg.npy',WB_sg)
np.save(SAVE_PATH+'WB_sg_mean.npy',WB_sg_mean)
np.save(SAVE_PATH+'WB_sg_std.npy',WB_sg_std)
 
np.save(SAVE_PATH+'WB_FK08_param.npy',WB_FK08_param)
np.save(SAVE_PATH+'Lf_Bodner_23.npy',Lf_Bodner_23)
np.save(SAVE_PATH+'Lf_FK_11.npy',Lf_FK_11)
 
# grid cell size for weighted average
np.save(SAVE_PATH+'drF.npy',drF)
 



