#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:18:20 2023

@author: javiera

climatologia gam
"""
import pandas as pd
import mikeio as mk
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

file = ("/home/javiera/Documents/IFOP/data/gam/"
        "gam_clim.dfsu")

layer_path = ("/home/javiera/Documents/IFOP/data/gam/"
              "mixed_layer_depth.nc")

def interp_value(idx, mixed_layer_depth):
    layer_aux = (mixed_layer_depth
                      .isel(time=0)
                          .to_dataframe()
                              .reset_index()
                                  .dropna()
                                      .reset_index(drop=True)
                )

    depth = layer_aux.depth[idx]
    
    profile = dfs.sel(x=layer_aux.lon[idx],
                y=layer_aux.lat[idx]).isel(time=0)
    
    df = pd.DataFrame(profile.geometry.element_coordinates,
                      columns=["lon", "lat", "z"]
                      )
    
    df["ac"] = profile.AC_Age_concentration.values
    
    df.sort_values(by="z",
                   ascending=False,
                   inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    
    arg = np.argmax(df.z <= -depth)
    
    df_aux = df.iloc[:arg].mean()
    
    interp_val = df_aux["ac"]
    
    return df_aux, interp_val


def ds_to_nc(dfs):
    
    nc = dfs.geometry.node_coordinates
    xn = nc[:,0]
    yn = nc[:,1]
    zn = nc[:,2]
    
    ec = dfs.geometry.element_coordinates
    xe = ec[:,0]
    ye = ec[:,1]
    ze = ec[:,2]
    
    # Time
    time = dfs.time
    
    # Node based data
    node_ids = list(range(len(nc)))
    z_dynamic = dfs._zn
    xn_da = xr.DataArray(xn,
                         coords=[node_ids],
                         dims=["nodes"],
                         attrs={'units': 'meter'}
                         )
    
    yn_da = xr.DataArray(xn,
                         coords=[node_ids],
                         dims=["nodes"],
                         attrs={'units': 'meter'}
                         )
    
    zn_da = xr.DataArray(zn,
                         coords=[node_ids],
                         dims=["nodes"],
                         attrs={'units': 'meter'}
                         )
    
    z_dyn_da = xr.DataArray(z_dynamic,
                            coords =[time,node_ids],
                            dims=["time", "nodes"],
                            attrs={'units': 'meter'}
                            )
    
    # Element based data
    el_ids = list(range(len(ec)))
    xe_da = xr.DataArray(xe,
                         coords=[el_ids],
                         dims=["elements"],
                         attrs={'units': 'meter'}
                         )
    
    ye_da = xr.DataArray(ye,
                         coords=[el_ids],
                         dims=["elements"],
                         attrs={'units': 'meter'}
                         )
    
    ze_da = xr.DataArray(ze,
                         coords=[el_ids],
                         dims=["elements"],
                         attrs={'units': 'meter'}
                         )
    
    # Add coordinates for nodes and elements
    data_dict = {'x': xn_da,
                 'y' :yn_da,
                 'z' : zn_da,
                 'xe' : xe_da,
                 'ye' : ye_da,
                 'ze' : ze_da,
                 'z_dynamic' : z_dyn_da}
    
    # add rest of data
    for da in dfs:
            da = xr.DataArray(da.to_numpy(), 
                              coords = [time,el_ids],
                              dims=["time", "elements"],
                              attrs={'units': da.unit.name})
    
            data_dict[da.name] = da
    
    
    # Here are some examples of global attributes, which is useful, but in most cases not required
    attributes={"title" : "Golfo Almirante Montt",
                "project": "Escalas",
                "source": "Mike 3 FM",
                "institution": "IFOP"}
    
    # create an xarray dataset
    xr_ds = xr.Dataset(data_dict,
                       attrs=attributes)
    return xr_ds

dfs = mk.read(file)
# %%

# mixed_layer_depth = xr.open_dataset(layer_path)

# layer_aux = (mixed_layer_depth
#                  .isel(time=0)
#                      .to_dataframe()
#                          .reset_index()
#                              .dropna()
#                                  .reset_index(drop=True)
#             )

# idx = 5
# prof = layer_aux.depth[idx]
# a = dfs.sel(x=layer_aux.lon[idx],
#             y=layer_aux.lat[idx]).isel(time=0)

# df = pd.DataFrame(a.geometry.element_coordinates,
#                   columns=["lon", "lat", "z"]
#                   )

# df["ac"] = a.AC_Age_concentration.values

# df.sort_values(by="z",
#                ascending=False,
#                inplace=True)

# df.reset_index(drop=True, inplace=True)

# arg = np.argmax(df.z <= -prof)

# fig, ax = plt.subplots(dpi=300)
# (dfs.sel(x=layer_aux.lon[10],
#         y=layer_aux.lat[10])["AC, Age concentration"]
#             .plot(extrapolate=False,
#                   marker='o',
#                   ax=ax)
# )

