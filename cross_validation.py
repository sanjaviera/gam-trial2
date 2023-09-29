#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:05:57 2023

@author: javiera
cross-validation
"""

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import xarray as xr
import pandas as pd
import numpy as np

path = ("/home/javiera/Documents/IFOP/data/")
dom = "gam"

df = pd.read_csv(path+dom+"/data_"+dom+".csv",
                 sep=" ")
df.where(df.yr > 2020).dropna().reset_index(drop=True)
obs = df.groupby(["lon", "lat"]).apply(len).shape[0]

cols = df.prof.max()
df_out = pd.DataFrame(np.zeros(shape=(obs, cols)))
df_out.values[:] = np.nan

# Define una función personalizada para guardar todo el DataFrame en una lista


def custom_agg(df_group):
    return [df_group.reset_index(drop=True)]


# Agrupa por "lon" y "lat" y guarda los DataFrames completos en listas
result_df = df.groupby(["lon", "lat"]).apply(custom_agg).reset_index(drop=True)

idx = result_df.iloc[0][0][["prof", "temp"]]
for row in range(obs):
    # a = result_df.iloc[row][0][["prof","temp"]]
    a = result_df.iloc[row][0][["temp"]]

    df_out.iloc[row, :len(a)] = a.squeeze()

# %%

from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform

# Supongamos que tienes tus datos en ctd_data y la variable objetivo en target
# ctd_data es una matriz donde las filas son perfiles CTD y las columnas son profundidades
# target es la variable objetivo correspondiente
ctd_data = df_out.iloc[:200, :200]


# Crea un imputador que reemplace NaN con la media de la columna
imputer = SimpleImputer(strategy='mean')

# Aplica la imputación a tus datos
ctd_data_imputed = imputer.fit_transform(ctd_data)
# Define el número de divisiones para la validación cruzada

# Define el número de divisiones para la validación cruzada
n_splits = 100

# Crea un objeto KFold para dividir los datos por perfil CTD
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Inicializa una lista para almacenar los largos de correlación
correlation_lengths = []

# Realiza la validación cruzada
for train_index, test_index in kf.split(ctd_data_imputed):
    train_profiles, test_profiles = ctd_data_imputed[train_index], ctd_data_imputed[test_index]

    # Calcula las distancias euclidianas entre todas las pares de profundidades
    distances = pdist(train_profiles.T, 'euclidean')

    # Convierte las distancias en una matriz de distancias
    distance_matrix = squareform(distances)

    # Calcula la autocorrelación espacial
    autocorrelation = 1.0 - np.corrcoef(train_profiles, rowvar=False)

    # Encuentra el largo de correlación
    max_distance = 0.0
    for i in range(len(train_profiles)):
        for j in range(i, len(train_profiles)):
            if autocorrelation[i, j] <= 0.5:  # Ajusta el umbral según sea necesario
                max_distance = max(max_distance, distance_matrix[i, j])

    correlation_lengths.append(max_distance)

# Calcula el largo de correlación promedio a lo largo de todas las divisiones de la validación cruzada
average_correlation_length = np.mean(correlation_lengths)
print("Largo de correlación promedio:", average_correlation_length)

