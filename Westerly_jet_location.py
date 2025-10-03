import numpy as np
import xarray as xr
import pandas as pd
import os,sys
from datetime import datetime
import matplotlib.pyplot as plt
import intake

cloudpath=["https://www.dkrz.de/s/intake"]
poolpath="/pool/data/Catalogs/dkrz_cmip6_disk.json"
col = intake.open_esm_datastore("/work/ik1017/Catalogs/dkrz_cmip6_disk.json")


CMIPmodel = { 
      1:'TaiESM1',
      2:'AWI-ESM-1-1-LR',
      3:'AWI-ESM-1-REcoM',
      4:'BCC-CSM2-MR',
      5:'BCC-ESM1',
      6:'FGOALS-f3-L',
      7:'FGOALS-g3',
      8:'CanESM5',
      9:'IITM-ESM',
      10:'CNRM-CM6-1',
      11:'CNRM-CM6-1-HR',
      12:'CNRM-ESM2-1',
      13:'ACCESS-CM2',
      14:'EC-Earth3',
      15:'MPI-ESM-1-2-HAM',
      16:'INM-CM4-8',
      17:'INM-CM5-0',
      18:'IPSL-CM6A-LR',
      19:'MIROC6',
      20:'MPI-ESM1-2-HR',
      21:'MPI-ESM1-2-LR',
      22:'MRI-ESM2-0',
      23:'GISS-E2-1-G',
      24:'GISS-E2-1-G',
      25:'CESM2',
      26:'CESM2-FV2',
      27:'CESM2-WACCM',
      28:'CESM2-WACCM-FV2',
      29:'NorESM2-LM',
      30:'NorESM2-MM',
      31:'GFDL-CM4'
      }

CMIPinst = { 
      1:'AS-RCEC',
      2:'AWI',
      3:'AWI',
      4:'BCC',
      5:'BCC',
      6:'CAS',
      7:'CAS',
      8:'CCCma',
      9:'CCCR-IITM',
      10:'CNRM-CERFACS',
      11:'CNRM-CERFACS',
      12:'CNRM-CERFACS',
      13:'CSIRO-ARCCSS',
      14:'EC-Earth-Consortium',
      15:'HAMMOZ-Consortium',
      16:'INM',
      17:'INM',
      18:'IPSL',
      19:'MIROC',
      20:'MPI-M',
      21:'MPI-M',
      22:'MRI',
      23:'NASA-GISS',
      24:'NASA-GISS',
      25:'NCAR',
      26:'NCAR',
      27:'NCAR',
      28:'NCAR',
      29:'NCC',
      30:'NCC',
      31:'NOAA-GFDL'
      }

CMIPvariant = { 
      1:'r1i1p1f1',
      2:'r1i1p1f1',
      3:'r1i1p1f1',
      4:'r1i1p1f1',
      5:'r1i1p1f1',
      6:'r1i1p1f1',
      7:'r1i1p1f1',
      8:'r1i1p1f1',
      9:'r1i1p1f1',
      10:'r1i1p1f2',
      11:'r1i1p1f2',
      12:'r1i1p1f2',
      13:'r1i1p1f1',
      14:'r1i1p1f1',
      15:'r1i1p1f1',
      16:'r1i1p1f1',
      17:'r1i1p1f1',
      18:'r1i1p1f1',
      19:'r1i1p1f1',
      20:'r1i1p1f1',
      21:'r1i1p1f1',
      22:'r1i1p1f1',
      23:'r1i1p1f1',
      24:'r1i1p1f2',
      25:'r1i1p1f1',
      26:'r1i1p1f1',
      27:'r1i1p1f1',
      28:'r1i1p1f1',
      29:'r1i1p1f1',
      30:'r1i1p1f1',
      31:'r1i1p1f1'
      }


def jet_lat_quadratic_interpolation(u1d):
    # Returns the lat of the jet using a quadratic interpolation
    # interpolation on pre-averaged (time and space) field
    # u1d: an xarray with the zonal and temporal mean of ua
    jet_lat_first_guess = u1d.argmax(dim="lat").values
    lat_1d = u1d.lat
    x_interp = [lat_1d.isel(lat=jet_lat_first_guess-2).values -0, 
    lat_1d.isel(lat=jet_lat_first_guess-1).values -0,
    lat_1d.isel(lat=jet_lat_first_guess).values -0, 
    lat_1d.isel(lat=jet_lat_first_guess+1).values -0, 
    lat_1d.isel(lat=jet_lat_first_guess+2).values -0] 

    y_interp = [u1d.isel(lat=jet_lat_first_guess-2).values -0, 
    u1d.isel(lat=jet_lat_first_guess-1).values -0,
    u1d.isel(lat=jet_lat_first_guess).values -0, 
    u1d.isel(lat=jet_lat_first_guess+1).values -0, 
    u1d.isel(lat=jet_lat_first_guess+2).values -0] 

    reg = np.poly1d(np.polyfit(x_interp, y_interp, 2)) 
    max = -reg[1]/(2*reg[2]) # solution of f'(x) = 0 for a second degree polynomia
    return(max)

def find_jet_lat_cmip_model(i, file_path_for_saving):
    # i: index of the CMIP model
    # file_path_for_saving: CMIP6 jet location will be saved to this location
    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_model__' + str(i) + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:
        query = dict(
            source_id      = CMIPmodel[i], # selection of the model
            variable_id    = "ua",          # minimum temperature
            table_id       = "Amon",             # daily frequency
            experiment_id  = "historical",          # choose an experiment
            member_id      = CMIPvariant[i],       # choose a member ("r" realization, "i" initialization, "p" physics, "f" forcing)
        )

        model_name = CMIPmodel[i]
        # Intake looks for the query we just defined in the catalog of the CMIP6 data pool at DKRZ
        cat_reduced = col.search(**query)
        cmip_mod = cat_reduced[cat_reduced.keys()[0]].to_dask()
        ua_850 = cmip_mod.ua.sel(plev=85000, method='nearest').mean('lon').squeeze() # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(1980, 2015)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp.sel(time=ua_850.time.dt.month.isin([1, 11, 12])).idxmax(dim="lat").mean("time").values- 0
        jet_lat_annual_1 = ua_850_interp.idxmax(dim="lat").mean("time").values-0


        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        # jet_lat_summer_2 = jet_lat_quadratic_interp_on_native_grid(ua_850_interp_fesom_summer)
        # jet_lat_annual_2 = jet_lat_quadratic_interp_on_native_grid(ua_850_interp_fesom)
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850_interp.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850_interp.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850_interp.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)

        pd_i = pd.DataFrame([model_name, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
            columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_i.to_csv(path_to_jet_lat_files)
        print(i)        




def find_jet_lat_cmip_model_exceptions(i, file_path_for_saving):
    # Sometimes, error when loading some CMIP6 models - here's a workaround
    file_path_for_saving = file_path + 'jet_lat_model__' + str(i) + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:

        query = dict(
            source_id      = CMIPmodel[i], # selection of the model
            variable_id    = "ua",          # minimum temperature
            table_id       = "Amon",             # daily frequency
            member_id      = CMIPvariant[i],  # model
            experiment_id  = "historical"   # choose a member ("r" realization, "i" initialization, "p" physics, "f" forcing)
        )

        model_name = CMIPmodel[i]
        # Intake looks for the query we just defined in the catalog of the CMIP6 data pool at DKRZ
        cat_reduced = col.search(**query)
        # a = cat_reduced[cat_reduced.keys()[0]].to_dask()

        multiple_fp = cat_reduced[cat_reduced.keys()[0]].df.uri[0][:-17] + '*.nc'


        cmip_mod_850 = xr.open_mfdataset(multiple_fp).ua.sel(plev=85000, method='nearest').mean('lon')
        ua_850 = cmip_mod_850.sel(time=cmip_mod_850.time.dt.year.isin(np.arange(1980, 2015)))
        ua_850_summer = cmip_mod_850.where(cmip_mod_850.lat < -10).where(cmip_mod_850.lat>-75).dropna(dim='lat').sel(time=cmip_mod_850.time.dt.month.isin([1, 11, 12]))# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850 = cmip_mod_850.where(cmip_mod_850.lat < -10).where(cmip_mod_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_summer_interp = ua_850_summer.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_summer_interp.idxmax(dim="lat").mean("time").values- 0
        jet_lat_annual_1 = ua_850_interp.idxmax(dim="lat").mean("time").values-0



        # Method2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        # jet_lat_summer_2 = jet_lat_quadratic_interp_on_native_grid(ua_850_interp_fesom_summer)
        # jet_lat_annual_2 = jet_lat_quadratic_interp_on_native_grid(ua_850_interp_fesom)
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850_interp.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850_interp.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850_interp.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)

        pd_i = pd.DataFrame([model_name, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_i.to_csv(path_to_jet_lat_files)


# Loop through the CMIP models
# Impossible for i=2, 6, 14 due to data structure

path_for_saving = ...

for i in range(1, len(CMIPmodel) + 1):
    print(i)
    # Kernel crashes for i=2, 14
    # Probably because there are too many files to load at once
    # The data are separated by year
    try:
        find_jet_lat_cmip_model(i, path_for_saving)
    except:
        print('It is an exception: ' + str(i))
        find_jet_lat_cmip_model_exceptions(i, path_for_saving)


# ERA5=============================================================================================================================

def find_jet_lat_ERA5(file_path_era5, file_path_for_saving):
    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_ERA5_1980-2014'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:

        indir = file_path_era5
        ds= xr.open_mfdataset(''+indir+'ERA5_850hPa_u_1M_*_regular.nc', concat_dim="time", combine="nested",
                        data_vars='minimal', coords='minimal', compat='override')
        ds['time'] = pd.to_datetime(ds['time'].values, format='%Y%m%d') # Convert time axis to readable values
        ds

        ua_850 = ds.sel(time=ds.time.dt.year.isin(np.arange(1980, 2015))).u.mean(dim="lon", skipna=True)#  zonal mean and selection 0f 1980-2014
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_era5_annual = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_era5_summer = ua_850_interp_era5_annual.sel(time=ua_850_interp_era5_annual.time.dt.month.isin([11, 12, 1]))
        del ua_850

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = (ua_850_interp_era5_summer.squeeze().idxmax(dim="lat")).mean('time').values- 0
        jet_lat_annual_1 = (ua_850_interp_era5_annual.squeeze().idxmax(dim="lat")).mean('time').values- 0


        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        # jet_lat_summer_2 = jet_lat_quadratic_interp_on_native_grid(ua_850_interp_fesom_summer)
        # jet_lat_annual_2 = jet_lat_quadratic_interp_on_native_grid(ua_850_interp_fesom)
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_era5_annual.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([ERA5, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2, jet_lat_summer_3, jet_lat_annual_3]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2',})
        pd_.to_csv(path_to_jet_lat_files)


file_path_era5 = ...
file_path_for_saving = ...
find_jet_lat_ERA5(file_path_era5, file_path_for_saving)


# EERIE models=====================================================================================================
# IFS-FESOM
# Hist

def find_jet_lat_fesom_hist(file_path_for_saving):
    model = 'ifs-fesom2-sr'
    expid =  'hist-1950'
    version = 'v20240304'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = '3D_monthly_avg'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_1980-2014'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:
        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()

        ua_850 = ds.mu.sel(level=85000, method='nearest').groupby('lat').mean() # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(1980, 2014)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_fesom = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_fesom_summer = ua_850_interp_fesom.sel(time=ua_850_interp_fesom.time.dt.month.isin([11, 12, 1])) #.mean("time")
        # ua_850_interp_fesom = ua_850_interp_fesom.mean("time")
        # del ua_850

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_fesom_summer.idxmax(dim="lat").mean('time').values- 0
        jet_lat_annual_1 = ua_850_interp_fesom.idxmax(dim="lat").mean('time').values-0


        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_fesom.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)

file_path_for_saving = ...
find_jet_lat_fesom_hist(file_path_for_saving)

# Spinup



def find_jet_lat_fesom_spinup(file_path_for_saving):

    model = 'ifs-fesom2-sr'
    expid =  'eerie-spinup-1950'
    version = 'v20240304'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = 'daily_3d'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_1980-2014'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:


        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds

        ua_850 = ds.u.sel(plev=85000, method='nearest').mean('lon') # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_fesom = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_fesom_summer = ua_850_interp_fesom.sel(time=ua_850_interp_fesom.time.dt.month.isin([11, 12, 1]))
        ua_850_interp_fesom = ua_850_interp_fesom
        # del ua_850

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_fesom_summer.idxmax(dim="lat").mean('time').values- 0
        jet_lat_annual_1 = ua_850_interp_fesom.idxmax(dim="lat").mean('time').values-0

        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_fesom.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)


file_path_for_saving = ...
find_jet_lat_fesom_spinup(file_path_for_saving)

# Control

def find_jet_lat_fesom_control(file_path_for_saving):
    model = 'ifs-fesom2-sr'
    expid =  'eerie-control-1950'
    version = 'v20240304'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = '3D_monthly_avg'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_1980-2014'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:


        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds = ds.rename({'value':'latlon'}).set_index(latlon=("lat","lon")).unstack("latlon")

        ua_850 = ds.mu.sel(level=85000, method='nearest').mean('lon') # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(1950, 2014)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_fesom = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_fesom_summer = ua_850_interp_fesom.sel(time=ua_850_interp_fesom.time.dt.month.isin([11, 12, 1]))
        ua_850_interp_fesom = ua_850_interp_fesom


        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_fesom_summer.idxmax(dim="lat").mean('time').values- 0
        jet_lat_annual_1 = ua_850_interp_fesom.idxmax(dim="lat").mean('time').values-0

        print('Method 1 done')

        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_fesom.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)

file_path_for_saving = ...
find_jet_lat_fesom_control(file_path_for_saving)


# ICON
# Hist

def find_jet_lat_icon_control(file_path_for_saving):
    model = 'icon-esm-er'
    expid =  'hist-1950'
    version = 'v20240618'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = 'plev19_monthly_mean'

    file_path = "/home/b/b383336/data/paper_SAM_persistence/"
    path_to_jet_lat_files = file_path + 'jet_lat_' + model + '-' + expid + '_1950-2014'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:


        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds

        ua_850 = ds.ua.sel(plev_2=85000, method='nearest').mean('lon') # selection of pressure level at 850 hPa and zonal mean
        # ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(1950, 1971)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_icon = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_icon_summer = ua_850_interp_icon.sel(time=ua_850_interp_icon.time.dt.month.isin([11, 12, 1]))

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_icon_summer.idxmax(dim="lat").mean("time").values- 0
        jet_lat_annual_1 = ua_850_interp_icon.idxmax(dim="lat").mean("time").values-0


        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_icon.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)

file_path_for_saving = ...
find_jet_lat_icon_control(file_path_for_saving)

# Spinup

def find_jet_lat_icon_spinup(file_path_for_saving):
    model = 'icon-esm-er'
    expid =  'eerie-spinup-1950'
    version = 'v20240618'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = 'plev19_1mth_mean'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_1980-1990'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:


        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds

        ua_850 = ds.ua.sel(plev_2=85000, method='nearest').mean('lon') # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(1980, 1991)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_icon = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_icon_summer = ua_850_interp_icon.sel(time=ua_850_interp_icon.time.dt.month.isin([11, 12, 1]))
        # ua_850_interp_icon = ua_850_interp_icon.mean("time")
        # del ua_850

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_icon_summer.idxmax(dim="lat").mean("time").values- 0
        jet_lat_annual_1 = ua_850_interp_icon.idxmax(dim="lat").mean("time").values-0

        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_icon.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)

file_path_for_saving = ...
find_jet_lat_icon_spinup(file_path_for_saving)

# Control

def find_jet_lat_icon_control(file_path_for_saving):
    model = 'icon-esm-er'
    expid =  'eerie-control-1950'
    version = 'v20240618'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = 'plev19_daily_mean'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_2041-2060'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:


        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds

        ua_850 = ds.ua.sel(plev_2=85000, method='nearest').mean('lon') # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(2040, 2061)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_icon = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_icon_summer = ua_850_interp_icon.sel(time=ua_850_interp_icon.time.dt.month.isin([11, 12, 1])) #.mean("time")
        # ua_850_interp_icon = ua_850_interp_icon.mean("time")
        # del ua_850

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_icon_summer.idxmax(dim="lat").mean("time").values- 0
        jet_lat_annual_1 = ua_850_interp_icon.idxmax(dim="lat").mean("time").values-0

        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_icon.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)

file_path_for_saving = ...
find_jet_lat_icon_control(file_path_for_saving)

# IFS AMIP=======================================================================================================================================
# AMIP9

def find_jet_lat_AMIP9_hist(file_path_for_saving):
    model =  'ifs-amip-tco1279'
    expid =  'hist.v20240901.atmos.gr025'
    version = 'v20240618'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = '3D_monthly'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_1980-2023'  + '.csv'
    if os.path.exists(path_to_jet_lat_files):
        print(path_to_jet_lat_files + "already exists")
    else:

        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")

        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds

        ua_850 = ds.avg_u.sel(level=850, method='nearest').groupby('lat').mean() # selection of pressure level at 850 hPa and zonal mean
        ua_850 = ua_850.sel(time=ua_850.time.dt.year.isin(np.arange(1980, 2024)))
        ua_850 = ua_850.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')# Selection of the range where to look for the jet maximum (between 10deg S and 75)
        ua_850_interp_amip9 = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
        ua_850_interp_amip9_summer = ua_850_interp_amip9.sel(time=ua_850_interp_amip9.time.dt.month.isin([11, 12, 1]))
        # del ua_850

        # Method 1: Detect the jet with the maximum of zonal wind on 1deg grid
        jet_lat_summer_1 = ua_850_interp_amip9_summer.idxmax(dim="lat").mean("time").values- 0
        jet_lat_annual_1 = ua_850_interp_amip9.idxmax(dim="lat").mean("time").values-0

        # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
        # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
        jet_lat_annual_2_np = []
        jet_lat_summer_2_np = []
        for tt in ua_850_interp_amip9.time:
            if tt.time.dt.month.isin([11, 12, 1]):
                jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))
            else:
                jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_1deg_grid(ua_850.sel(time=tt).squeeze()))

        jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
        jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)

        pd_= pd.DataFrame([model + '-' + expid, jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
        columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
        pd_.to_csv(path_to_jet_lat_files)

file_path_for_saving = ...
find_jet_lat_AMIP9_hist(file_path_for_saving)

# IFS AMIP 28 (5 members)

def find_jet_lat_AMIP28_hist(file_path_for_saving):
    model =  'ifs-amip-tco399'
    expid =  'hist.v20240901.atmos.gr025'
    version = 'v20240618'
    realm = 'atmos'
    gridspec = 'gr025'
    fq = '3D_monthly'

    path_to_jet_lat_files = file_path_for_saving + 'jet_lat_' + model + '-' + expid + '_1980-2023'  + '.csv'

    if os.path.exists(path_to_jet_lat_files):
        print('File exists already')
    else:
        cat = intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")
        # #=========================================================
        cat_regrid = cat['dkrz.disk.model-output'][model][expid]
        print('Data frequency :',list(cat_regrid))
        # #=========================================================
        
        # cat_regrid = cat['dkrz.disk.model-output'][model][expid][version][realm][gridspec]
        ds = cat_regrid[fq].to_dask()
        ds
        
        
        # Step 1: save subsets, otherwise it is too heavy to process
        for real in ds.realization:
            if os.path.exists(file_path + '/to_delete_later/ua850_ifs-amip-tco399_real_' + str(i) + '.nc4'):
                print(str(real.item()) + ' exists')
            else:
                ua_850_real = ds.sel(realization=i).sel(level=850, method='nearest').avg_u.groupby('lat').mean() 
                ua_850 = ua_850_real.where(ua_850.lat < -10).where(ua_850.lat>-75).dropna(dim='lat')
                ua_850.to_netcdf(file_path + '/to_delete_later/ua850_ifs-amip-tco399_real_' + str(real.item()) + '.nc4')



        # Compute for each realization the jet lat
        
        pd_real = []
        for real in ds.realization:
            print(str(real.item()) + ' start')
            ua_850 = xr.open_dataset(file_path + '/to_delete_later/ua850_ifs-amip-tco399_real_' + str(real.item()) + '.nc4').avg_u
            ua_850_interp_amip9 = ua_850.interp(lat = np.linspace(-75, -10, 66)) # linear interpolation onto a regular 1deg grid
            ua_850_interp_amip9_summer = ua_850_interp_amip9.sel(time=ua_850_interp_amip9.time.dt.month.isin([11, 12, 1]))
            
            jet_lat_summer_1 = ua_850_interp_amip9_summer.idxmax(dim="lat").mean("time").values- 0
            jet_lat_annual_1 = ua_850_interp_amip9.idxmax(dim="lat").mean("time").values-0
            print('method 1 done')
        
        
            # Method 2: Detect the jet with the quadratic maximum of zonal wind on the native grid
            # Quadratic interpolation = 2nd degree interpolation using 2 grid points before and after the spotted maximum
            jet_lat_annual_2_np = []
            jet_lat_summer_2_np = []
            for tt in ua_850_interp_amip9.time:
                if tt.time.dt.month.isin([11, 12, 1]):
                    jet_lat_summer_2_np.append(jet_lat_quadratic_interp_on_native_grid(ua_850.sel(time=tt).squeeze()))
                    jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_native_grid(ua_850.sel(time=tt).squeeze()))
                else:
                    jet_lat_annual_2_np.append(jet_lat_quadratic_interp_on_native_grid(ua_850.sel(time=tt).squeeze()))
        
            jet_lat_annual_2 = sum(jet_lat_annual_2_np)/len(jet_lat_annual_2_np)
            jet_lat_summer_2 = sum(jet_lat_summer_2_np)/len(jet_lat_summer_2_np)
        
            print('method 2 done')
            pd_= pd.DataFrame([model + '-real:' + str(real.item()), jet_lat_summer_1, jet_lat_annual_1, jet_lat_summer_2, jet_lat_annual_2]).transpose().rename(
            columns={0:'Model', 1:'jet_lat_summer_1', 2:'jet_lat_annual_1', 3:'jet_lat_summer_2', 4:'jet_lat_annual_2'})
            pd_real.append(pd_)
            print(str(real.item()) + ' end')
        
        pd_ = pd.concat(pd_real, axis=0)
        pd_.to_csv(path_to_jet_lat_files)


file_path_for_saving = ...
find_jet_lat_AMIP28_hist(file_path_for_saving)