#!/usr/bin/python3

"""
This program computes and plots the PDP in CMIP6 models. This is done through
time-extended EOF analysis of 850mb geopotential height anomalies over the
North Pacific after removing variability associated with Pacific Decadal
Oscillation.

MUST BE EXECUTED IN CONDA XESMF ENVIRONMENT
"""

from netCDF4 import MFDataset, num2date
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm
from scipy import stats
import xarray as xr
import imageio
from dask.distributed import Client,LocalCluster,as_completed
import dask
import zarr
import gcsfs
import xesmf as xe
import xscale.signal as xs
import pandas as pd
from eofs.standard import Eof

# Implement a dask local cluster
cluster = LocalCluster(n_workers=10, threads_per_worker=1)
client = Client(cluster)

dask.config.set({"array.slicing.split_large_chunks": True})

# User-created functions
from butter_bandpass import butter_bandpass_filter # applies bandpass filter given highcut and lowcut
from EOF import EOF_func # EOF analysis using SVD method
from linreg import linreg # Simple linear regression. Returns slope and predicted value
from EEOF import EEOF_func # EEOF analysis using SVD method

################################################################################
# USER INPUT -- SELECT FILES
def set_vars():
    lvl = 8.5e+04
    var = 'zg'
    lowcut = 20. # Lowcut of bandpass filter
    highcut = 7. # Highcut of bandpass filter

    return lvl,var,lowcut,highcut
################################################################################
def regrid_1x1(ds):

# Create new grid

    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90.0,91.0,1)),
                         'lon': (['lon'], np.arange(0.0,360.0,1))
                        }
                       )
# Build Regridder
    regridder = xe.Regridder(ds,ds_out,'bilinear')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def xr_reshape(A, dim, newdims, coords):
    """ Reshape DataArray A to convert its dimension dim into sub-dimensions given by
    newdims and the corresponding coords.
    Example: Ar = xr_reshape(A, 'time', ['year', 'month'], [(2017, 2018), np.arange(12)]) """


    # Create a pandas MultiIndex from these labels
    ind = pd.MultiIndex.from_product(coords, names=newdims)

    # Replace the time index in the DataArray by this new index,
    A1 = A.copy()

    A1.coords[dim] = ind

    # Convert multiindex to individual dims using DataArray.unstack().
    # This changes dimension order! The new dimensions are at the end.
    A1 = A1.unstack(dim)

    # Permute to restore dimensions
    i = A.dims.index(dim)
    dims = list(A1.dims)

    for d in newdims[::-1]:
        dims.insert(i, d)

    for d in newdims:
        _ = dims.pop(-1)


    return A1.transpose(*dims)
################################################################################
def load_hgt(df,lvl):
    """
    Load data for the given source and interpolate to a 1x1 grid. Then isolate
    time domain to extended boreal winter seasons and spatial domain to
    the north Pacific. Returns numpy arrays with mean computed over each
    extended boreal winter season.
    """
    print('Loading geopotential height data...')
    ds = {};
    for source_id in tqdm(df['source_id']):
        try:
            vad = df[(df.source_id==source_id)].zstore.values[0]

            gcs = gcsfs.GCSFileSystem(token='anon')
            ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True)
            ds[source_id] = regrid_1x1(ds[source_id])

            # grab lats/lons
            lon = ds[source_id].lon.sel(lon=slice(125,270)).values
            lat = ds[source_id].lat.sel(lat=slice(15,80)).values

            # isolate in time and space
            ds[source_id] = ds[source_id].zg.sel(plev = lvl,method='nearest')
            ds[source_id] = ds[source_id].sel(lon = slice(125,270),lat = slice(15,80))
            idx = np.where((ds[source_id].time.dt.month>=11) | (ds[source_id].time.dt.month<=3))[0][3:-2]
            ds[source_id] = ds[source_id][idx,:,:]

            # compute anomalies
            tlen = ds[source_id].shape[0]
            mon = 5 # Number of months for extended boreal winter (5 for Nov - Mar)
            year = int(tlen/mon) # Number of years in the trimmed data set

            climatology = ds[source_id].groupby("time.month").mean("time")
            ds[source_id] = (ds[source_id].groupby("time.month") - climatology).drop('month')
            ds[source_id] = xr_reshape(ds[source_id], 'time', ['season count', 'month'], [np.arange(year), [11,12,1,2,3]])
            ds[source_id] = ds[source_id].groupby('season count').mean('month')
        except:
            pass
    return ds,lat,lon
################################################################################
def remove_trend(ds):
    print('removing linear trends...')
    for source_id in tqdm(ds):
        xs.fitting.detrend(ds[source_id], dim='season count', type='linear')
        ds[source_id] = ds[source_id].values
    return ds
################################################################################
def bpass_filter(ds,lowcut,highcut):
    print('filtering...')
    filtered_zg={};
    for source_id in ds:
        filtered_zg[source_id] = butter_bandpass_filter(ds[source_id].reshape(ds[source_id].shape[0],lat.size*lon.size),lowcut,highcut,1)
    return filtered_zg
################################################################################
def remove_PDO(ds,lat,lon):
    print('Removing PDO variability...')
    coslat = np.reshape(np.repeat(lat,lon.size,axis=0),(lon.size,lat.size)).T
    wgts = np.reshape(np.sqrt(np.cos(np.deg2rad(coslat))),(lat.size*lon.size)) #weighting
    for source_id in ds:
        solver = Eof(ds[source_id],weights=wgts)
        PDO = solver.reconstructedField(neofs=1) # variability associated with first eof
        ds[source_id] = ds[source_id]-PDO
    return ds
################################################################################
def compute_eeof(ds,lat,lon):
    # Set up time-lagged matrix
    lag = 6 # number of lags (0 through 5, so 6 total)
    coslat = np.reshape(np.repeat(lat,lon.size,axis=0),(lon.size,lat.size)).T
    wgts = np.reshape(np.sqrt(np.cos(np.deg2rad(coslat))),(lat.size*lon.size)) #weighting
    da = {}
    print('Computing EEOFs...')
    for source_id in tqdm(ds):
        eeof_wgts = wgts
        length = ds[source_id].shape[0]-5 # used to determine indices for formatting matrix for EEOF analysis
        space = lat.size*lon.size # spatial domain
        eeof_matrix = ds[source_id][:length,:]

        for t in range(5):
            eeof_matrix = np.concatenate((ds[source_id][t:length+t,:],eeof_matrix),axis=1)
            eeof_wgts = np.concatenate((eeof_wgts,wgts),axis=0)

            solver = Eof(eeof_matrix,weights=eeof_wgts)
            pc1 = np.squeeze(solver.pcs(npcs=1, pcscaling=1))
            regressed_EEOF = np.ones((space,lag))*np.nan # initializing regression maps matrix
        for t in range (0,lag): # creating regression maps
            regressed_EEOF[:,t] = linreg(pc1,ds[source_id][t:(length+t),:])[0]
        da[source_id] = regressed_EEOF
    return da
################################################################################
def plot_maps(ds,lat,lon):
    for source_id in ds:
        llcrnrlon = lon[0] # Setting bounds for maps
        llcrnrlat = lat[0]
        urcrnrlon = lon[-1]
        urcrnrlat = lat[-1]
        labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']



        EEOF_range = np.arange(-10,10.5,0.5)
        figure = plt.figure(figsize=(8,8))
        for i in range(6):
            row = int(i/2)
            col = int(i%2)
            n = -2 + i
            plt.subplot2grid((3,2),(row,col),colspan=1)
            map = Basemap(projection='mill', llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,\
                urcrnrlat=urcrnrlat, resolution='l')
            x, y = map(*np.meshgrid(lon,lat))
            map.drawcoastlines(color='black')
                # if i == 0:
                #     tran = map.plot(lonlon,latlat,latlon = True,color='k',markersize = 0.5)
                #     lonpts = [lonlon[0],lonlon[408],lonlon[816],lonlon[1222]]
                #     latpts = [latlat[0],latlat[408],latlat[816],latlat[1222]]
                #     lonpts,latpts = map(lonpts,latpts)
                #     pts = map.plot(lonpts,latpts,'ko')
                #     tlabels = ['A','B','C','D']
                #     plt.text(lonpts[0],latpts[0]-1.5e6,tlabels[0],fontsize = fontsize, color = 'k')
                #     plt.text(lonpts[1]+5e5,latpts[1],tlabels[1],fontsize = fontsize, color = 'k')
                #     plt.text(lonpts[2],latpts[2]+3e5,tlabels[2],fontsize = fontsize, color = 'k')
                #     plt.text(lonpts[3]-1.5e6,latpts[3],tlabels[3],fontsize = fontsize, color = 'k')
            # nlevs = 255 # 250 for left of zero
            cmap = plt.cm.seismic
            cs = plt.contourf(x,y,np.reshape(ds[source_id][:,i],(lat.size,lon.size)),\
                EEOF_range,cmap = plt.cm.seismic,extend='both')
            # cmap._lut[nlevs/2-6:nlevs/2+7] = [1,1,1,1]
            plt.title(labels[i]+' Year '+str(n),fontsize = 16)
            cbar_ax = figure.add_axes([0.87, 0.03, 0.03, 0.92])
            cb = figure.colorbar(cs, cax=cbar_ax,orientation='vertical',label='(m)',ticks = [-10,-5,0,5,10])
            cb.set_label(label='(m)', size='x-large')
            cb.ax.tick_params(labelsize='x-large')
            figure.subplots_adjust(right=0.84, top = 0.96, left = 0.08, bottom = 0.02, wspace = 0.05, hspace = 0.4)
            plt.savefig('/home/disk/rocinante/rawrgers/Figures/PDP/CMIP6/'+source_id+'.png',dpi=400)
################################################################################

def main():
        # Set all variables
        lvl,var,lowcut,highcut = set_vars()

        # Grab 850mb height data from the cloud
        df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        zg,lat,lon = client.submit(load_hgt,df.query("activity_id=='CMIP' & experiment_id=='piControl' & member_id=='r1i1p1f1' & table_id=='Amon' & variable_id=='zg' & grid_label=='gn'"),lvl).result()

        # Remove linear trend
        zg = remove_trend(zg)

        # Filter heights and remove PDO variability
        filtered_zg = bpass_filter(zg,lowcut,highcut) # implement filter
        filtered_zg = remove_PDO(filtered_zg,lat,lon)

        # Compute eeof
        eeofs = compute_eeof(filtered_zg,lat,lon)

        # Plot and save maps!
        plot_maps(eeofs,lat,lon)

if __name__ == "__main__":
