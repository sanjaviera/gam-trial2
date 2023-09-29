#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:41:40 2023

@author: javiera
"""
import xarray as xr
path_out = ("/home/javiera/Documents/IFOP/data/julia/tojulia")
file = ("/home/javiera/Documents/IFOP/data/grillas/integrate_patagonia.nc")
bat = xr.open_dataset(file)

bat = bat.rename({"longitude":"lon",
                  "latitude":"lat",
                  "z":"bat"})

bat_dom = bat

tol = -20 # m
bat_dom.bat.values = bat_dom.bat.values + tol

bat_dom.to_netcdf(f"{path_out}/bathymetry_2julia.nc")