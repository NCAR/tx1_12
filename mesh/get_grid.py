import numpy as np
import netCDF4
import xarray as xr


def getgrid(supergrid,topography):

    """
    This function takes a user-defined supergrid and creates an xarray dataset 
    containing the metrics of the corresponding model grid.

    Modified from /glade/work/gmarques/cesm/tx1_12/mesh/gen_nc_grid.py
    by Ben Barr (bbarr@ucar.edu).

    Parameters:
        supergrid - path + filename for supergrid.
        topography - path + filename for topography.  Only used for 'depth' and 'tmask'.
            May be set to None if these fields are not needed.
    Return:
        ds - an xarray dataset containing the following model grid metrics:
            'tlon','tlat' - lon/lat of t point locations [deg]
            'ulon','ulat' - lon/lat of u point locations [deg]
            'vlon','vlat' - lon/lat of v point locations [deg]
            'qlon','qlat' - lon/lat of q point locations [deg]
            'tarea' - t cell area [m2]
            'angle' - t-point angle [deg]
            'dxt' - x-distance between u-points, centered at t [m]
            'dyt' - y-distance between v-points, centered at t [m]
            'dxCv' - x-distance between q-points, centered at v [m]
            'dyCu' - y-distance between q-points, centered at u [m]
            'dxCu' - x-distance between t-points, centered at u [m]
            'dyCv' - y-distance between t-points, centered at v [m]
            'ar' - grid aspect ratio [-]
            'egs' - grid effective grid spacing [deg]
            if topography is not None:
                'tmask' - ocean fraction at t-cell centers [-]
                'depth' -  depth at h points [m]
    """


    # 1. Import supergrid and topography -------------------

    # Get MOM6 supergrid
    nc_sgrd = netCDF4.Dataset(supergrid)
    x = nc_sgrd.variables['x'][:]    # [deg]
    y = nc_sgrd.variables['y'][:]    # [deg]
    dx = nc_sgrd.variables['dx'][:]    # [m]
    dy = nc_sgrd.variables['dy'][:]    # [m]
    area = nc_sgrd.variables['area'][:]    # [m2]
    angle_dx = nc_sgrd.variables['angle_dx'][:]    # [deg]
    nc_sgrd.close()

    # Get topography and mask
    if topography is not None:
        nc_topo = netCDF4.Dataset(topography)
        depth = nc_topo.variables['D_interp'][:]    # [m]
        tmask = nc_topo.variables['mask'][:]    # [-]
        # Or create the mask ad hoc
        #tmask = np.where(depth >= 5, 1, 0)
        nc_topo.close()


    # 2. Calculate grid metrics ----------------------

    # T point locations
    tlon = x[1::2,1::2]
    tlat = y[1::2,1::2]

    # U point locations
    ulon = x[1::2,::2]
    ulat = y[1::2,::2]

    # V point locations
    vlon = x[::2,1::2]
    vlat = y[::2,1::2]

    # Corner point locations
    qlon = x[::2,::2]
    qlat = y[::2,::2]

    # T cell area (sum of four supergrid cells)
    # Note: The original file gen_nc_grid.py has a bug here.  Below line is correct.
    tarea = area[::2,::2] + area[1::2,1::2] + area[::2,1::2] + area[1::2,::2]

    # t-point angle
    angle = angle_dx[1::2,1::2]

    # x-distance between u-points, centered at t
    dxt = dx[1::2,::2] + dx[1::2,1::2]

    # y-distance between v-points, centered at t
    dyt = dy[::2,1::2] + dy[1::2,1::2]

    # x-distance between  q-points, centered at v
    dxCv = dx[2::2,::2] + dx[2::2,1::2]

    # y-distance between  q-points, centered at u
    dyCu = dy[::2,2::2] + dy[1::2,2::2]

    # x-distance between t-points, centered at u
    dxCu = dx[1::2,1::2] + np.roll(dx[1::2,1::2], -1, axis=-1)

    # y-distance between t-points, centered at v
    dyCv = dy[1::2,1::2] + np.roll(dy[1::2,1::2], -1, axis=0)

    # grid aspect ratio
    ar = dyt / dxt

    # grid effective grid spacing
    # A = 4*pi*r^2 , area of sphere of radius r
    # dA = (r*cos(theta)*dlambda)*(r*dtheta), differential area on sphere
    #    = r^2*domega
    # domega = dA/r^2, differential solid angle  (steradians, sr)
    # 1 sr = (180./pi)^2 square degrees
    costheta = np.cos(tlat*np.pi/180.)
    rearth = 637122000 # Earth radius in centimeter
    domega = tarea / rearth**2
    egs  = np.sqrt(domega * (180./np.pi)**2)


    # 3. Create dataset --------------------------------------

    ds = xr.Dataset()

    ds['tlon']  = xr.DataArray(data=tlon , name='tlon' , dims=['ny' ,'nx'] , attrs={'long_name':'array of t-grid longitudes','units':'degrees_east'})
    ds['tlat']  = xr.DataArray(data=tlat , name='tlat' , dims=['ny' ,'nx'] , attrs={'long_name':'array of t-grid latitudes','units':'degrees_north'})
    ds['ulon']  = xr.DataArray(data=ulon , name='ulon' , dims=['ny' ,'nxp'], attrs={'long_name':'array of u-grid longitudes','units':'degrees_east'})
    ds['ulat']  = xr.DataArray(data=ulat , name='ulat' , dims=['ny' ,'nxp'], attrs={'long_name':'array of u-grid latitudes','units':'degrees_north'})
    ds['vlon']  = xr.DataArray(data=vlon , name='vlon' , dims=['nyp','nx'] , attrs={'long_name':'array of v-grid longitudes','units':'degrees_east'})
    ds['vlat']  = xr.DataArray(data=vlat , name='vlat' , dims=['nyp','nx'] , attrs={'long_name':'array of v-grid latitudes','units':'degrees_north'})
    ds['qlon']  = xr.DataArray(data=qlon , name='qlon' , dims=['nyp','nxp'], attrs={'long_name':'array of q-grid longitudes','units':'degrees_east'})
    ds['qlat']  = xr.DataArray(data=qlat , name='qlat' , dims=['nyp','nxp'], attrs={'long_name':'array of q-grid latitudes','units':'degrees_north'})
    ds['dxt']   = xr.DataArray(data=dxt  , name='dxt'  , dims=['ny' ,'nx'] , attrs={'long_name':'x-distance between u-points, centered at t','units':'meters'})
    ds['dyt']   = xr.DataArray(data=dyt  , name='dyt'  , dims=['ny' ,'nx'] , attrs={'long_name':'y-distance between v-points, centered at t','units':'meters'})
    ds['dxCv']  = xr.DataArray(data=dxCv , name='dxCv' , dims=['ny' ,'nx'] , attrs={'long_name':'x-distance between  q-points, centered at v','units':'meters'})
    ds['dyCu']  = xr.DataArray(data=dyCu , name='dyCu' , dims=['ny' ,'nx'] , attrs={'long_name':'y-distance between  q-points, centered at u','units':'meters'})
    ds['dxCu']  = xr.DataArray(data=dxCu , name='dxCu' , dims=['ny' ,'nx'] , attrs={'long_name':'x-distance between  t-points, centered at u','units':'meters'})
    ds['dyCv']  = xr.DataArray(data=dyCv , name='dyCv' , dims=['ny' ,'nx'] , attrs={'long_name':'y-distance between  t-points, centered at v','units':'meters'})
    ds['tarea'] = xr.DataArray(data=tarea, name='tarea', dims=['ny' ,'nx'] , attrs={'long_name':'area of t-cells','units':'meters^2'})
    ds['angle'] = xr.DataArray(data=angle, name='angle', dims=['ny' ,'nx'] , attrs={'long_name':'angle grid makes with latitude line','units':'degrees'})
    ds['ar']    = xr.DataArray(data=ar   , name='ar'   , dims=['ny' ,'nx'] , attrs={'long_name':'grid aspect ratio (dyt/dxt)','units':'none'})
    ds['egs']   = xr.DataArray(data=egs  , name='egs'  , dims=['ny' ,'nx'] , attrs={'long_name':'grid effective grid spacing','units':'degrees'})
    if topography is not None:
        ds['tmask'] = xr.DataArray(data=tmask, name='tmask', dims=['ny' ,'nx'] , attrs={'long_name':'ocean fraction at t-cell centers','units':'none'})
        ds['depth'] = xr.DataArray(data=depth, name='depth', dims=['ny' ,'nx'] , attrs={'long_name':'depth at h points','units':'meters'})

    return ds



