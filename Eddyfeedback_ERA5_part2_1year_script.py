#!/usr/bin/env python

import numpy as np
import xarray as xr
import pandas as pd
import os,sys
from datetime import datetime
from scipy.signal import detrend

import warnings
warnings.filterwarnings("ignore")

import psutil


print("Argument 1:", sys.argv[1])
yyyy= sys.argv[1]

### Read in information about the SAM index and EOF (based on u)

ds_sam = xr.open_dataset("ERA5_1980_2023_SAMindex_u_S20.nc")

coslat = np.cos(np.deg2rad(ds_sam.coords['lat'].values)).clip(0., 1.)

### For eddy momentum flux convergence calculation

indir='/work/bk1377/b382037/WP7/ERA5/' #before1979

### U file 

ds_u = xr.open_mfdataset(''+indir+'ERA5_250_500_700_850hPa_u_6H_'+yyyy+'_regular.nc', concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override')

ds_u['time'] = pd.to_datetime(ds_u['time'].values, format='%Y%m%d')

#condition_reg = (ds_u['lat'] <= 0)
condition_reg = (ds_u['lat'] > -90) & (ds_u['lat'] <= -20)
# Apply the condition to select data within the latitude range
computed_condition_reg = condition_reg.compute()
ds_u = ds_u.where(computed_condition_reg, drop=True)

ds_u = ds_u.isel(plev=[0,1,3])

### V file

ds_v = xr.open_mfdataset(''+indir+'ERA5_250_500_700_850hPa_v_6H_'+yyyy+'_regular.nc', concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override')

ds_v['time'] = pd.to_datetime(ds_v['time'].values, format='%Y%m%d')

ds_v = ds_v.where(computed_condition_reg, drop=True)

ds_v = ds_v.isel(plev=[0,1,3])

# To zonal averaging first

uaz = ds_u.mean(dim="lon", skipna=True)
vaz = ds_v.mean(dim="lon", skipna=True)

momflux= ((ds_u['u'] - uaz['u'])*(ds_v['v'] - vaz['v'])).mean(dim="lon", skipna=True)

del vaz
ds_u.close()
ds_v.close()

# Cos latitude square 
cos_sq=np.square(np.cos(np.deg2rad(momflux.coords['lat'].values)))
coords_lat= {"lat":momflux.coords['lat'],}
cossq_xr = xr.DataArray(cos_sq, dims=["lat"], coords= coords_lat)

momflux_phi= momflux* cossq_xr
momflux_phi['lat']= momflux_phi['lat']* np.pi / 180 # convert deg to radian
cossq_xr['lat']= cossq_xr['lat']* np.pi / 180
a=6371*1.e3 #meter

### It is unclear in Simpson et al. (2013) when the momentum flux is converted to daily means 
### Let's test the results if daily mean is taken first before taking the differential. 

momflux_phi_daily = momflux_phi.resample(time='D').mean()


EPFdiv_daily = momflux_phi_daily.differentiate("lat")
#EPFdiv = momflux_phi.differentiate("lat")


del momflux_phi

EPFdivterm_daily = EPFdiv_daily/cossq_xr*(-1/a)
#EPFdivterm = EPFdiv/cossqr_xr*(-1/a)


### Convert to daily means 
uaz_da = uaz['u'].resample(time='D').mean()

### Do pressure-weighted vertical average : 
# sum (A*p*dp) /sum (p*dp)
# But after testing, I think Simpson et al. (2013) used the following:
# pressure-weighted vertical average : sum (A*dp) /sum (dp) 

pweights = xr.DataArray(
    data=np.array([250,300,250]),
    dims=["plev"])
pweights.name = "weights"

EPFdivterm_weimean  = EPFdivterm_daily.weighted(pweights).mean(dim='plev')
uaz_weimean = uaz_da.weighted(pweights).mean(dim='plev')

del EPFdiv_daily
del pweights
del EPFdivterm_daily
del uaz

#uaz_weimean = ds_sam['u_sam']
uaz_weimean['lat']= uaz_weimean['lat']* np.pi / 180


### Convert to daily means 
#EPFdivterm_daily = EPFdivterm_weimean.resample(time='D').mean()
#uaz_daily = uaz_weimean.resample(time='D').mean()

EPFdivterm_daily = EPFdivterm_weimean.chunk({'time': 366, 'lat': 320})
uaz_daily_new = uaz_weimean.chunk({'time': 366, 'lat': 320})

### Projecting onto the SAM eof 
### (1) EPFdivterm

wgts_diag=np.diag(coslat)
EOF1_u=ds_sam['SAMeof'].to_numpy()
denom = np.sqrt(np.matmul(np.matmul(EOF1_u,wgts_diag),EOF1_u.T))

We=np.matmul(wgts_diag, EOF1_u.T)

coords_lat= {"lat":EPFdivterm_daily.coords['lat'],}
We_xrarray = xr.DataArray(We.flatten(), dims=["lat"], coords= coords_lat)

EPFdiv_sam = xr.dot(EPFdivterm_daily, We_xrarray, dims="lat")/denom

### (2) u1D term 

#timethisyear=EPFdivterm_daily.coords['time']
#u_2D= ds_sam['u_sam'].sel(time=timethisyear)

#u1D_sam= xr.dot(uaz_daily, We_xrarray, dims="lat")/denom
u1D_sam_new= xr.dot(uaz_daily_new, We_xrarray, dims="lat")/denom

### Writing out files 

#Eddyfeedbackterms= EPFdiv_sam.to_dataset(name='eddyforcing')
#Eddyfeedbackterms['u1D'] = u1D_sam

Eddyfeedbackterms_new= EPFdiv_sam.to_dataset(name='eddyforcing')
Eddyfeedbackterms_new['u1D'] = u1D_sam_new

#print(u1D_sam.values)
#outputFileName = './new/ERA5_SAMeddyforcingstrength_'+yyyy+'_S20.nc'
outputFileName = './new/ERA5_SAMeddyforcingstrength_'+yyyy+'_S20_new.nc'

print ('saving to ', outputFileName)
Eddyfeedbackterms_new.to_netcdf(path=outputFileName)
print ('finished saving')

del EPFdiv_sam
del u1D_sam_new
del Eddyfeedbackterms_new

#Occasionally, xarray objects retain lazy-loaded data, leading to unexpected results when saving. Try copying the data explicitly to force evaluation.
