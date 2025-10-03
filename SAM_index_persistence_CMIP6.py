#!/usr/bin/env python

import numpy as np
import xarray as xr
import pandas as pd
#import intake
import os,sys
from datetime import datetime

# For processing data

from scipy.signal import detrend
from scipy.fftpack import rfft, irfft, fftfreq, fft
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from eofs.standard import Eof
from eofs.examples import example_data_path
import math 

### For plotting

import matplotlib.pyplot as plt

#-----------------------------------


#####################
# Read in data & setup
#####################

print("Argument 1:", sys.argv[1])
print("Argument 2:", sys.argv[2])

infile   = sys.argv[1] #'/gws/nopw/j04/eerie/model_output/MOHC/HadGEM3-GC5-EERIE-HH/piControl/day/zg_500/'
ds      = xr.open_dataset(infile)
varin   ='zg'         

outdir = '/work/bk1377/b382037/WP7/CMIP6/'     

EXPname = sys.argv[2]

print('EXPname:', EXPname)
#-----------


ds['time'] = pd.to_datetime(ds['time'].values, format='ISO8601')
ds = ds.sel(time=slice("1980-01-01","2014-12-31"))
total_years = 35           # Total years of the intended analysis 

ds2 = ds[varin].to_dataset() 
ds2 = ds2.rename_vars({varin: 'z'})
ds.close()

# (1) Do zonal averaging first
zbar = ds2.mean(dim="lon", skipna=True)


# (2) Substract the global mean (following the method in Gerber et al., 2010)
z_globe = zbar.mean(dim="lat", skipna=True)
daily_value = zbar - z_globe
daily_value = daily_value.squeeze()

# (3) Detrend the data
detrended = detrend(daily_value['z'], axis=0, type='linear')


# (4) Deseasonalize the data: 
# (4a) First, apply a 60-day low-pass filter
nt=detrended.shape[0]

t=np.arange(0, nt)
print(t.shape)

xf=fft(detrended, axis=0)

f_signal = rfft(detrended, axis=0)
w = fftfreq(detrended.shape[0], d=t[1]-t[0])

cut_f_signal = f_signal.copy()
cut_f_signal[(np.abs(w)>1/60)] = 0 
lowpass60 = irfft(cut_f_signal, axis=0) 

daily_value["z_detrended"]=(['time', 'lat'],  detrended.astype(float))
daily_value["z_detrended_60d"]=(['time', 'lat'],  lowpass60.astype(float))


# (4b) Second, apply a apply a 30-year low-pass filter to the smoothed time series, 
# using only that calendar date of each year in the data

def func(group):
    
    x=group.to_numpy()
    nt=x.shape[0]
    t=np.arange(0, nt)
    xf=fft(x, axis=0)
    f_signal = rfft(x, axis=0)
    w = fftfreq(x.shape[0], d=t[1]-t[0])

    cut_f_signal = f_signal.copy()
    cut_f_signal[(np.abs(w)>1/30)] = 0 
    cut_signal = irfft(cut_f_signal, axis=0)

    group["z_detrended_30y"]=(['time', 'lat'],  cut_signal.astype(float))

    return group["z_detrended_30y"]


clim_test= daily_value["z_detrended_60d"].groupby("time.dayofyear").apply(func)


# (5) Obtain the final anomalies (internal variability is the “noise”)

anom_final= daily_value["z_detrended"]-clim_test["z_detrended_30y"]
anom = anom_final.reset_coords(drop=True)


#--- Compute the leading EOF for the region south of 20°S


# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.

z = anom.sel(lat=slice(-89.9,-20))
znumpy=z.to_numpy()


coslat = np.cos(np.deg2rad(z.coords['lat'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(znumpy, weights=wgts)

eof1 = solver.eofs(neofs=1)
pc1   = solver.pcs(npcs=1)
pc1_rescaled  = solver.pcs(npcs=1, pcscaling=1)
eof1_rescaled = solver.eofs(neofs=1, eofscaling=2) 
variance_fractions = solver.varianceFraction(neigs=1)

#--- Obtain the SAM index time series (PC1) 


# I want to define the positive phase of SAM as low-pressure anomalies over high latitudes:

if (eof1[0,0]<0):  # The point value at the south pole
    pc1_ar= pc1_rescaled.squeeze()    
else:    
    pc1_ar= pc1_rescaled.squeeze()*-1
    eof1_rescaled= eof1_rescaled*-1

pc1_ar.reshape((-1, 1))


# ### Calculate the e-folding timescale
# 
# The diagnostic used is the decorrelation timescale, which is the e‐folding timescale of the autocorrelation function of the SAM index.
# 
# take a 180 day window around a given day, smoothing it with a Gaussian filter with a full width at half maximum (FWHM) of 60 days
# calculate the autocorrelation of above smoothened SAM index


coords= {'time':daily_value["z"].coords['time'],}
SAM_window    = xr.DataArray(np.full((51,nt),fill_value=np.nan), dims=["lags","time"], coords= coords)

for i in range(0, nt-51):  # Adjust indices to account for slicing

       # Extract the 180-day window around the current day
        window_data = pc1_ar[i:i+51]    
        SAM_window.isel(time=i).values[:]    = window_data



# For simplicity, remove 29 Feb

SAM_window_noleap=SAM_window.convert_calendar('noleap')
mean_SAM = SAM_window_noleap.groupby("time.dayofyear").mean("time")
mean_SAM_np = mean_SAM.to_numpy()
SAM_window_noleap_np = SAM_window_noleap.to_numpy()



coords= {'time':SAM_window_noleap.isel(time=slice(0,365)).coords['time'],}
ACF_ens = xr.DataArray(np.full((51,365),fill_value=np.nan), dims=["lags","time"], coords= coords)


for days in range(0,365):
    for lags in range(51): 
        numerator = 0.
        denominatorA = 0.
        denominatorB = 0.
        
        for yy in range(total_years):             
            if math.isnan(SAM_window_noleap_np[0,days+365*yy]) != True :
                numerator    =  numerator   + (SAM_window_noleap_np[0,days+365*yy]-mean_SAM_np[0,days])*(SAM_window_noleap_np[0+lags,days+365*yy]-mean_SAM_np[0+lags,days])
                denominatorA =  denominatorA + (SAM_window_noleap_np[0,days+365*yy]-mean_SAM_np[0,days])**2
                denominatorB =  denominatorB + (SAM_window_noleap_np[0+lags,days+365*yy]-mean_SAM_np[0+lags,days])**2  
         
        ACF_ens[lags, days] = numerator/((denominatorA*denominatorB)**0.5) 



# Following the method of Simpson et al. (2013):
# --- Apply Gaussian Filtering on the ACF (done over the year (of the same date), at each lag; not over lag)
# sigma parameter=18 is also used in Simpson et al. (2013)

ACF_ens_3year = xr.concat([ACF_ens,ACF_ens,ACF_ens],"time")
ACF_ens_smooth = gaussian_filter(ACF_ens_3year, sigma=18, axes=1)
ACF_ens_smooth = ACF_ens_smooth[:,365:365+365]


# Fitting to an exponential function to identify the e-folding timescale

def model_func(x, a, k):
    return a * np.exp(-k * x)
    
def model_func_re(y, a, k):
    return -1/k * np.log(y / a)

timescale= xr.DataArray(np.full(365,fill_value=np.nan))
fiterr= xr.DataArray(np.full(365,fill_value=np.nan))

inputx = np.arange(0, 51, dtype=float)

for i in range(0, 365):
    
    # Fit the model to the data
    inputy = ACF_ens_smooth[:,i]
    popt, pcov = curve_fit(model_func, inputx, inputy)
    
    y_fit = model_func(inputx, *popt)
    
    fiterr[i] = np.sqrt(np.mean((y_fit - inputy) ** 2))
    efoldtime = model_func_re(1/math.e,*popt)
    timescale[i] = efoldtime


shifted_timescale = np.roll(timescale, -181, axis=0)



#####################
# Write out data 
#####################

np.savetxt(outdir+EXPname+'_SAMtimescale.out', shifted_timescale, delimiter=',')
=


#####################
coords= {"time":daily_value["z"].coords['time'],}

SAMindex = xr.DataArray(
    data=pc1_ar,
    dims=["time"],
    coords= coords,
    name="SAMindex",
)

outputFileName = outdir+EXPname+'_SAMindex.nc'
SAMindex.to_netcdf(path=outputFileName)
SAMindex.close()

#####################
coords= {"lat":z.coords['lat'],}


SAMeof = xr.DataArray(
    data=eof1_rescaled[0,:],
    dims=["lat"],
    coords= coords,
    name="EOF1",
)

outputFileName2 = outdir+EXPname+'_SAMeof.nc'
#outputFileName2 = 'ERA5_1958_1978_SAMeof.nc'

SAMeof.to_netcdf(path=outputFileName2)
SAMeof.close()
ds2.close()
