#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:43:37 2023

@author: javiera

generate nc with observations to DIVAnd
"""
import pandas as pd
import xarray as xr

def obs2julia(fn, var, dom):
   """
   fn: path of the ctd observations
   var: (string) variable to adjust as nc file
   dom: (string) domain name    
   """
   path_save = ("/home/javiera/Documents/IFOP/data/julia/tojulia/")
   
   df = pd.read_csv(fn, sep=" ")
   df["station"] = (df.groupby(["lon","lat"]).ngroup().astype("S14"))
   df = df.dropna(subset=["prof", var])
   
   # Set dimensions and coordinates
   dims = ['observations']
   coords = {
       'obslon': ('observations', df['lon']),
       'obslat': ('observations', df['lat']),
       'obstime': ('observations', pd.to_datetime(df['date'])),
       'obsdepth': ('observations', df['prof'])
   }
   
   ds = xr.Dataset({var: (dims, df[var]),              
                    'obsid': (dims, df['station'])
                    }, coords=coords)

   ds.to_netcdf(path_save+dom+"_"+var+".nc")

# def concat_fromjulia():
    
   

#%%
if __name__ == '__main__':
    
    dom = "gam"
    fn = ("/home/javiera/Documents/IFOP/data/gam/data_"+dom+"_sin_outliers.csv")    
    df = pd.read_csv(fn, sep=" ")
    numerical = ['sal',
                 'temp',
                 # 'od_ml',
                 'dens',
                 'fl',
                 'od_mg',
                 'od_sat',
                 # 'od_mol_l',
                 # 'od_mol_kg',
                 # 'cond',
                 'nitra',
                 'nitri',
                 'fos', 
                 'sil',
                 'cl_tot',
                 'cl_act',
                 # 'feop' 
                ]

    for var in numerical:
        obs2julia(fn, var, dom)



    
    # fn_ref = ("/home/javiera/Downloads/WOD-Salinity.nc")
    # ds_ref = xr.open_dataset(fn_ref)
    
    # fn_obs = ("/home/javiera/Documents/IFOP/Escalas/2022-2023/data/"
    #           "chiloe_aysen_2010-2021/matriz_patagonianorte.csv")
    
    # df = pd.read_csv(fn_obs, 
    #                  sep=" ")
    # df = df.rename(columns={"yr":"year"})
    
    # df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # df = df.rename(columns={"year":"yr"})
    # df["station"] = df.groupby(["lon","lat"]).ngroup().astype("S14")
    
    # df = df.dropna(subset=["prof", "sal"])
    
    # # Set dimensions and coordinates
    # dims = ['observations']
    # coords = {
    #     'obslon': ('observations', df['lon']),
    #     'obslat': ('observations', df['lat']),
    #     'obstime': ('observations', df['date']),
    #     'obsdepth': ('observations', df['prof'])
    # }
    
    # # Drop the dimensions and coordinates columns from the DataFrame
    # # data_vars = df.drop(['longitude', 'latitude', 'time', 'depth'], axis=1)
    # # data_vars = df.sal
    # # Create the xarray Dataset
    
    # ds = xr.Dataset({'salinity': (dims, df['sal']),
    #                  'temperature': (dims, df["temp"]),
    #                  'oxygen': (dims, df["od_mg"]),
    #                  'nitra': (dims, df["nitra"]),
    #                  'fos': (dims, df["fos"]),
    #                  'sil': (dims, df["sil"]),                 
    #                  'obsid': (dims, df['station'])
    #                  }, coords=coords)
    
    # ds.to_netcdf("/home/javiera/Documents/IFOP/Escalas/"
    #               "2022-2023/source/julia/2julia/obs_test.nc")
    
    # print(ds)