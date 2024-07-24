#!/usr/bin/env python
# coding: utf-8

from xmitgcm import llcreader
model = llcreader.PleiadesLLC4320Model()
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int)
args = parser.parse_args()
print('My index is = ', args.index)

CASE_NAME = '01_gulf/'
BASE = '/nobackup/abodner/llc4320_data/'
PATH = BASE+CASE_NAME



# select subdomain
# depth above -700m (as we will average over the mixed layer below)
# lat/lon domain to be larger than 12 degrees

lat_min = 30
lat_max = 45
lon_min = -60
lon_max = -45
depth_lim = -700


print('T,S,UV,W FULL DATASETS FROM LLC4320')
print('CASE '+CASE_NAME)
print(datetime.now())
isExist = os.path.exists(PATH)
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(PATH)


isExist = os.path.exists(PATH+'raw_data/')
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(PATH+'raw_data/')


if args.index == 1:
	######################################
	############## T ####################
	######################################


	print('LOAD DATASETS: ds_T')
	print(datetime.now())
	ds_T_full = model.get_dataset(varnames=['Theta'], type='latlon')

	print('SELECET AREA ON T GRID')
	print(datetime.now())
	sel_area_T = np.logical_and(np.logical_and(np.logical_and(ds_T_full.XC>lon_min, ds_T_full.XC<lon_max ),
                           np.logical_and(ds_T_full.YC>lat_min, ds_T_full.YC<lat_max)), ds_T_full.Z>depth_lim)

	######################################
	############## T #####################
	######################################

	print('ds_T:SELECT')
	print(datetime.now())
	ds_T = ds_T_full.where(sel_area_T, drop=True).coarsen(time=12, boundary="trim").mean()

	print('ds_T:SAVE NETCDF')
	print(datetime.now())
	ds_T.Theta.to_netcdf(PATH+'raw_data/ds_T.nc',engine='h5netcdf')
	del ds_T, ds_T_full
	print('ds_T:COMPLETE')


if args.index == 2:
	######################################
	############## S #####################
	######################################

	print('LOAD DATASETS: ds_T,ds_S')
	print(datetime.now())
	ds_T_full = model.get_dataset(varnames=['Theta'], type='latlon')
	ds_S_full = model.get_dataset(varnames=['Salt'], type='latlon')

	print('SELECET AREA ON T GRID')
	print(datetime.now())
	sel_area_T = np.logical_and(np.logical_and(np.logical_and(ds_T_full.XC>lon_min, ds_T_full.XC<lon_max ),
                           np.logical_and(ds_T_full.YC>lat_min, ds_T_full.YC<lat_max)), ds_T_full.Z>depth_lim)

	print('ds_S:SELECT')
	print(datetime.now())
	ds_S = ds_S_full.where(sel_area_T, drop=True).coarsen(time=12, boundary="trim").mean()

	print('ds_S:SAVE NETCDF')
	print(datetime.now())
	ds_S.Salt.to_netcdf(PATH+'raw_data/ds_S.nc',engine='h5netcdf')
	del ds_S, ds_S_full, sel_area_T
	print('ds_S:COMPLETE')



elif args.index == 3:
	######################################
	############## UV ####################
	######################################

	print('LOAD DATASET: ds_UV')
	print(datetime.now())
	ds_UV_full = model.get_dataset(varnames=['U','V'], type='latlon')

	######################################
	############## U #####################
	######################################

	print('SELECET AREA ON U GRID')
	print(datetime.now())

	sel_area_U = np.logical_and(np.logical_and(np.logical_and(ds_UV_full.XG.mean('j_g')>lon_min, ds_UV_full.XG.mean('j_g')<lon_max),
                           np.logical_and(ds_UV_full.YC.mean('i')>lat_min, ds_UV_full.YC.mean('i')<lat_max)),ds_UV_full.Z>depth_lim)
	
	print('ds_U:SELECT')
	print(datetime.now())
	ds_U = ds_UV_full.U.where(sel_area_U, drop=True).coarsen(time=12, boundary="trim").mean()

	print('ds_U:SAVE NETCDF')
	print(datetime.now())
	ds_U.to_netcdf(PATH+'raw_data/ds_U.nc',engine='h5netcdf')
	del ds_U, sel_area_U
	print('ds_U:COMPLETE')



elif args.index == 4:
        ######################################
        ############## UV ####################
        ######################################

	print('LOAD DATASET: ds_UV')
	print(datetime.now())
	ds_UV_full = model.get_dataset(varnames=['U','V'], type='latlon')

	######################################
	############## V #####################
	######################################

	print('SELECET AREA ON V GRID')
	print(datetime.now())

	sel_area_V = np.logical_and(np.logical_and(np.logical_and(ds_UV_full.XC.mean('j')>lon_min, ds_UV_full.XC.mean('j')<lon_max),
                           np.logical_and(ds_UV_full.YG.mean('i_g')>lat_min, ds_UV_full.YG.mean('i_g')<lat_max)),ds_UV_full.Z>depth_lim)

	print('ds_V:SELECT')
	print(datetime.now())
	ds_V = ds_UV_full.V.where(sel_area_V, drop=True).coarsen(time=12, boundary="trim").mean()

	print('ds_V:SAVE NETCDF')
	print(datetime.now())
	ds_V.to_netcdf(PATH+'raw_data/ds_V.nc',engine='h5netcdf')
	del ds_V,  ds_UV_full, sel_area_V
	print('ds_V:COMPLETE')


elif args.index == 5:
	######################################
	############## W #####################
	######################################

	print('LOAD DATASET: ds_W')
	print(datetime.now())
	ds_W_full = model.get_dataset(varnames=['W'], type='latlon')

	print('SELECET AREA ON W GRID')
	print(datetime.now())

	sel_area_W = np.logical_and(np.logical_and(np.logical_and(ds_W_full.XC>lon_min, ds_W_full.XC<lon_max ),
                           np.logical_and(ds_W_full.YC>lat_min, ds_W_full.YC<lat_max)), ds_W_full.Zl>depth_lim)

	print('ds_W:SELECT')
	print(datetime.now())
	ds_W = ds_W_full.where(sel_area_W, drop=True).coarsen(time=12, boundary="trim").mean()


	print('ds_W:SAVE NETCDF')
	print(datetime.now())
	ds_W.W.to_netcdf(PATH+'raw_data/ds_W.nc',engine='h5netcdf')
	del ds_W, ds_W_full, sel_area_W
	print('ds_W:COMPLETE')



	######################################
	############## HBL ###################
	######################################


	print('LOAD DATASET: ds_HBL')
	print(datetime.now())
	ds_HBL_full = model.get_dataset(varnames=['KPPhbl'], type='latlon')

	print('SELECET AREA ON HBL GRID')
	print(datetime.now())

	sel_area_HBL = np.logical_and(np.logical_and(ds_HBL_full.XC>lon_min, ds_HBL_full.XC<lon_max ),np.logical_and(ds_HBL_full.YC>lat_min, ds_HBL_full.YC<lat_max))


	print('ds_HBL:SELECT')
	ds_HBL = ds_HBL_full.where(sel_area_HBL, drop=True).coarsen(time=12, boundary="trim").mean()


	print('ds_HBL:SAVE NETCDF')
	print(datetime.now())
	ds_HBL.KPPhbl.to_netcdf(PATH+'raw_data/ds_HBL.nc',engine='h5netcdf')
	del ds_HBL, ds_HBL_full, sel_area_HBL
	print('ds_HBL:COMPLETE')




	######################################
	############## Q #####################
	######################################



	
	print('LOAD DATASET: ds_Q')
	print(datetime.now())
	ds_Q_full = model.get_dataset(varnames=['oceQnet'], type='latlon')

	print('SELECET AREA ON Q  GRID')
	print(datetime.now())


	sel_area_Q = np.logical_and(np.logical_and(ds_Q_full.XC>lon_min, ds_Q_full.XC<lon_max ),np.logical_and(ds_Q_full.YC>lat_min, ds_Q_full.YC<lat_max))

	print('ds_Q:SELECT')
	ds_Q = ds_Q_full.where(sel_area_Q, drop=True).coarsen(time=12, boundary="trim").mean()


	print('ds_Q:SAVE NETCDF')
	print(datetime.now())
	ds_Q.oceQnet.to_netcdf(PATH+'raw_data/ds_Q.nc',engine='h5netcdf')
	del ds_Q, ds_Q_full, sel_area_Q
	print('ds_Q:COMPLETE')



	######################################
	############## TAU ###################
	######################################

	print('LOAD DATASET: ds_TAU')
	print(datetime.now())
	ds_TAU_full = model.get_dataset(varnames=['oceTAUX','oceTAUY'], type='latlon')


	######################################
	############## TAUX ###################
	######################################


	print('SELECET AREA ON TAUX GRID')
	print(datetime.now())


	sel_area_TAUX = np.logical_and(np.logical_and(ds_TAU_full.XG.mean('j_g')>lon_min, ds_TAU_full.XG.mean('j_g')<lon_max),
                                np.logical_and(ds_TAU_full.YC.mean('i')>lat_min, ds_TAU_full.YC.mean('i')<lat_max))

	print('ds_TAUX:SELECT')
	print(datetime.now())
	ds_TAUX = ds_TAU_full.oceTAUX.where(sel_area_TAUX, drop=True).coarsen(time=12, boundary="trim").mean()

	print('ds_TAUX:SAVE NETCDF')
	print(datetime.now())
	ds_TAUX.to_netcdf(PATH+'raw_data/ds_TAUX.nc',engine='h5netcdf')
	del ds_TAUX, sel_area_TAUX
	print('ds_TAUX:COMPLETE')




	######################################
	############## TAUY ###################
	######################################


	print('SELECET AREA ON TAUY GRID')
	print(datetime.now())


	sel_area_TAUY = np.logical_and(np.logical_and(ds_TAU_full.XC.mean('j')>lon_min, ds_TAU_full.XC.mean('j')<lon_max),
                                np.logical_and(ds_TAU_full.YG.mean('i_g')>lat_min, ds_TAU_full.YG.mean('i_g')<lat_max))


	print('ds_TAUY:SELECT')
	print(datetime.now())
	ds_TAUY = ds_TAU_full.oceTAUY.where(sel_area_TAUY, drop=True).coarsen(time=12, boundary="trim").mean()

	print('ds_TAUY:SAVE NETCDF')
	print(datetime.now())
	ds_TAUY.to_netcdf(PATH+'raw_data/ds_TAUY.nc',engine='h5netcdf')
	del ds_TAUY, sel_area_TAUY
	print('ds_TAUY:COMPLETE')



