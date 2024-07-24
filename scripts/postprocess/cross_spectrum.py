#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import xgcm 
from fastjmd95 import jmd95numba 
import glob


#path
BASE = '/scratch/ab10313/pleiades/'

PATH_LIST_2d = glob.glob(BASE+'*/2d_data/')
PATH_LIST_3d = glob.glob(BASE+'*/3d_data/')
PATH_LIST_PP = glob.glob(BASE+'*/preprcossed_data/')

for i_file in np.arange(0,len(PATH_LIST_PP)):
   
    # paths to dataset
    PATH_2d = PATH_LIST_2d[i_file]
    PATH_3d = PATH_LIST_3d[i_file]

    # make diirectory for preprocessed variables
    PATH_PP = PATH_LIST_PP[i_file]



    # load 3d data
    ds_T = xr.open_dataset(PATH_3d+'ds_T.nc',engine="h5netcdf")
    ds_S = xr.open_dataset(PATH_3d+'ds_S.nc',engine="h5netcdf")
    ds_W = xr.open_dataset(PATH_3d+'ds_W.nc',engine="h5netcdf")



    # find min and max i and j to crop to 10X10 degrees

    i_min = np.max([ds_T.i.min().values, ds_S.i.min().values, ds_W.i.min().values])
    i_max = np.min([ds_T.i.max().values, ds_S.i.max().values, ds_W.i.max().values])
    j_min = np.max([ds_T.j.min().values, ds_S.j.min().values, ds_W.j.min().values])
    j_max = np.min([ds_T.j.max().values, ds_S.j.max().values, ds_W.j.max().values])


    #define slice to 480 index

    if i_min+480>i_max:
        print('cropped region error in i')
    elif j_min+480>j_max:
        print('cropped region error in j')
    else:
        i_slice = slice(i_min,i_min+480)
    j_slice = slice(j_min,j_min+480)



    # merge datasets

    ds_3d =xr.merge([ds_T.sel(i=i_slice,j=j_slice), ds_S.sel(i=i_slice,j=j_slice), ds_W.sel(i=i_slice,j=j_slice)])


    # define grids 
    grid_3d = xgcm.Grid(ds_3d)


    # sigma from temp and salt, using the fastjmd95 package

    # reference density 
    rho0 = 1000 #kg/m^3

    # potential density anomaly 
    # with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
    sigma0 = jmd95numba.rho(ds_3d.Salt.chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size}),
                     ds_3d.Theta.chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size}), 0) - rho0

    # gravity
    G = 9.81 #m/s^2

    # buoyancy
    B = -G*sigma0/rho0
    B = B.rename('Buoyancy')


    del sigma0


    # interp W 
    W_interp = grid_3d.interp(ds_3d.W,'Z', boundary='extend')
    W_interp = W_interp.chunk(chunks={'time': 1, 'j': W_interp.j.size, 'i': W_interp.i.size, 'k': W_interp.k.size})

    del grid_3d, ds_3d


    # # cospectrum of w and b at the surface

    B_drop = B.drop(['CS', 'SN', 'Depth', 'dxF', 'dyF', 'rA', 'XC', 'YC','hFacC']).fillna(0)
    del B

    #  spectra
    import xrft
   
    WB_cross_spectra = xrft.isotropic_cross_spectrum(W_interp, B_drop, dim=['i','j'], 
                                           detrend='linear', window=True).compute().mean('time')


    #save spectra
    WB_cross_spectra.real.to_netcdf(PATH_PP+'WB_cross_spectra_z.nc',engine='h5netcdf')


    # save max over mixed layer depth average
    # load 3d data
    ds_T = xr.open_dataset(PATH_3d+'ds_T.nc',engine="h5netcdf")
    HML = xr.open_dataarray(PATH_PP+'HML.nc',engine="h5netcdf")

    k_HML = WB_cross_spectra.k.where(ds_T.Z<HML.mean()).min(dim="k",skipna=True)
    WB_spectra_mld = WB_cross_spectra.real.sel(k=slice(0,k_HML)).mean('k')

    dx = ds_T.dxF.mean()
    k_r = WB_cross_spectra.freq_r/dx/1e-3
    k_r_max = WB_spectra_mld.freq_r.where(WB_spectra_mld*k_r == (WB_spectra_mld*k_r).max(),drop=True)/dx/1e-3
    L_max = 1/k_r_max

    (WB_spectra_mld*k_r).to_netcdf(PATH_PP+'WB_kr_cross_spectra_mld.nc',engine='h5netcdf')
    k_r_max.to_netcdf(PATH_PP+'k_r_max_mld.nc',engine='h5netcdf')
