#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import xgcm 
from fastjmd95 import jmd95numba 



# paths to dataset
PATH_2d = '/scratch/ab10313/pleiades/01_gulf/2d_data/'
PATH_3d = '/scratch/ab10313/pleiades/01_gulf/3d_data/'

# make diirectory for preprocessed variables
PATH_PP = '/scratch/ab10313/pleiades/01_gulf/preprocessed_data/'
#os.mkdir(PATH_PP)



# load 2d data
ds_HBL = xr.open_dataset(PATH_2d+'ds_HBL.nc',engine="h5netcdf")
ds_Q = xr.open_dataset(PATH_2d+'ds_Q.nc',engine="h5netcdf")
ds_TAUX = xr.open_dataset(PATH_2d+'ds_TAUX.nc',engine="h5netcdf")
ds_TAUY = xr.open_dataset(PATH_2d+'ds_TAUY.nc',engine="h5netcdf")


# load 3d data
ds_T = xr.open_dataset(PATH_3d+'ds_T.nc',engine="h5netcdf")
ds_S = xr.open_dataset(PATH_3d+'ds_S.nc',engine="h5netcdf")
ds_U = xr.open_dataset(PATH_3d+'ds_U.nc',engine="h5netcdf")
ds_V = xr.open_dataset(PATH_3d+'ds_V.nc',engine="h5netcdf")
ds_W = xr.open_dataset(PATH_3d+'ds_W.nc',engine="h5netcdf")



# find min and max i and j to crop to 10X10 degrees

i_min = np.max([ds_HBL.i.min().values,ds_Q.i.min().values, ds_TAUX.i_g.min().values, ds_TAUY.i.min().values,
                ds_T.i.min().values, ds_S.i.min().values, ds_U.i_g.min().values, ds_V.i.min().values, ds_W.i.min().values])


i_max = np.min([ds_HBL.i.max().values,ds_Q.i.max().values, ds_TAUX.i_g.max().values, ds_TAUY.i.max().values,
                ds_T.i.max().values, ds_S.i.max().values, ds_U.i_g.max().values, ds_V.i.max().values, ds_W.i.max().values])


j_min = np.max([ds_HBL.j.min().values,ds_Q.j.min().values, ds_TAUX.j.min().values, ds_TAUY.j_g.min().values,
                ds_T.j.min().values, ds_S.j.min().values, ds_U.j.min().values, ds_V.j_g.min().values, ds_W.j.min().values])


j_max = np.min([ds_HBL.j.max().values,ds_Q.j.max().values, ds_TAUX.j.max().values, ds_TAUY.j_g.max().values,
                ds_T.j.max().values, ds_S.j.max().values, ds_U.j.max().values, ds_V.j_g.max().values, ds_W.j.max().values])



#define slice to 480 index

if i_min+480>i_max:
    print('cropped region error in i')
elif j_min+480>j_max:
    print('cropped region error in j')
else:
    i_slice = slice(i_min,i_min+480)
    j_slice = slice(j_min,j_min+480)




# merge datasets
ds_2d =xr.merge([ds_HBL.sel(i=i_slice,j=j_slice), ds_Q.sel(i=i_slice,j=j_slice),
                 ds_TAUX.sel(i_g=i_slice,j=j_slice), ds_TAUY.sel(i=i_slice,j_g=j_slice)])


ds_3d =xr.merge([ds_T.sel(i=i_slice,j=j_slice), ds_S.sel(i=i_slice,j=j_slice),
                 ds_U.sel(i_g=i_slice,j=j_slice), ds_V.sel(i=i_slice,j_g=j_slice), ds_W.sel(i=i_slice,j=j_slice)])



# define grids 

grid_2d = xgcm.Grid(ds_2d)

grid_3d = xgcm.Grid(ds_3d)




# sigma from temp and salt, using the fastjmd95 package

    
# reference density 
rho0 = 1000 #kg/m^3

# potential density anomaly 
# with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
sigma0 = jmd95numba.rho(ds_3d.Salt.chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size}),
                         ds_3d.Theta.chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size}), 0) - rho0

sigma0 = sigma0.rename('sigma0')




# sigma0 at 10m depth for reference
sigma0_10m = sigma0.isel(k=6).broadcast_like(sigma0).chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size})
delta_sigma = sigma0 - sigma0_10m
del sigma0_10m



# gravity
G = 9.81 #m/s^2

# buoyancy
B = -G*sigma0/rho0
B = B.rename('Buoyancy')

# save buoyancy averaged over mixed layer depth:
B.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'B.nc',engine='h5netcdf')


# vertical buoyancy gradient (stratification) note the minus sign because z is negative
Nsquared = -B.diff(dim='k')/B.drF
Nsquared.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'Nsquared.nc',engine='h5netcdf')
del Nsquared

# horizontal x buoyancy gradient
B_x = B.diff(dim='i')/B.dxF
B_x.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'B_x.nc',engine='h5netcdf')


# horizontal y buoyancy gradient
B_y = B.diff(dim='j')/B.dyF
B_y.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'B_y.nc',engine='h5netcdf')
del B_y



# mixed layer depth
HML = sigma0.Z.broadcast_like(sigma0).where(delta_sigma<=0.03).min(dim="k",skipna=True).chunk(chunks={'time': 1, 'j': sigma0.j.size, 'i': sigma0.i.size}).rename('Mixed Layer Depth')
HML.to_netcdf(PATH_PP+'HML.nc',engine='h5netcdf')
del HML, sigma0




# interp velocities and buoyancy fluxes, average over MLD
U_interp = grid_3d.interp(ds_3d.U,'X', boundary='extend')
U_interp.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'U.nc',engine='h5netcdf')

UB = U_interp * B
UB.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'UB.nc',engine='h5netcdf')

del U_interp, UB


V_interp = grid_3d.interp(ds_3d.V,'Y', boundary='extend')
V_interp.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'V.nc',engine='h5netcdf')

VB = V_interp * B
VB.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'VB.nc',engine='h5netcdf')

del V_interp, VB


W_interp = grid_3d.interp(ds_3d.W,'Z', boundary='extend')
W_interp.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'W.nc',engine='h5netcdf')

WB = W_interp * B
WB.where(delta_sigma<=0.03).mean(dim="k",skipna=True).to_netcdf(PATH_PP+'WB.nc',engine='h5netcdf')

del W_interp, WB

# lat lon
lat = ds_2d.YC.mean('i')
lon = ds_2d.XC.mean('j')

lat.to_netcdf(PATH_PP+'lat.nc',engine='h5netcdf')
lon.to_netcdf(PATH_PP+'lon.nc',engine='h5netcdf')

# Coriolis 
omega = 7.2921e-5

FCOR = xr.zeros_like(B.isel(time=0,k=0))

for ii in range(len(lat)):
        FCOR[ii,:] = 2*omega*np.sin(lat[ii]* np.pi / 180.)

FCOR.to_netcdf(PATH_PP+'FCOR.nc',engine='h5netcdf')

del FCOR, lat, lon, B



# interp tau
TAUX_interp = grid_2d.interp(ds_2d.oceTAUX,'X', boundary='extend')
TAUX_interp.to_netcdf(PATH_PP+'TAUX.nc',engine='h5netcdf')

TAUY_interp = grid_2d.interp(ds_2d.oceTAUY,'Y', boundary='extend')
TAUY_interp.to_netcdf(PATH_PP+'TAUY.nc',engine='h5netcdf')


# save HBL
ds_2d.KPPhbl.to_netcdf(PATH_PP+'HBL.nc',engine='h5netcdf')


# save Q
ds_2d.oceQnet.to_netcdf(PATH_PP+'Q.nc',engine='h5netcdf')


