#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:51:05 2023

@author: javiera

GAM/outliers
"""
import pandas as pd
from data_analysis import grafica_perfil, visualize_outliers, VariableInfo
import numpy as np

path = ("/home/javiera/Documents/IFOP/data/")
path_figs = ("/home/javiera/Documents/IFOP/fig/python/gam/")

dom = "gam"

df = pd.read_csv(path+dom+"/data_"+dom+".csv",
                 sep=" ")
df = df.where(df.yr >  2020).dropna(how="all").reset_index(drop=True)
# # z_scores, z_scores_abs, df_filtered = visualize_outliers(df, 5)

# nitrato
idx = df.where(df.prof==1).nitra.argmax()
grafica_perfil(df,idx,"nitra")
df["nitra"][idx] = np.nan

# od_mg
grafica_perfil(df, 6596, "od_mg")
df["od_mg"][6596] = np.nan
df["od_ml"][6596] = np.nan
df["od_mol_kg"][6596] = np.nan
df["od_mol_l"][6596] = np.nan

df.to_csv(path+dom+"/data_"+dom+"_sin_outliers.csv", sep=" ", index=False)
#%%