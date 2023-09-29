#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:05:29 2023

@author: javiera

Extrae campos medios estacionales de WA en GAM
"""

import mikeio as mk
from mikeio import generic
import os

def list_files(carpeta, extension):
    archivos = []

    # Itera a través de todos los archivos y carpetas en la carpeta especificada
    for ruta, _, archivos_en_carpeta in os.walk(carpeta):
        for archivo in archivos_en_carpeta:
            if archivo.endswith(extension):
                archivos.append(os.path.join(ruta, archivo))

    return archivos



dom = "gam"
year = 2018
year2 = year+1
file = (f"/media/javiera/Expansion/ATLAS_GAM/WA_{year}.dfsu")

dfs = mk.Dfsu(file)
time = dfs.time.drop(f"{year2}-01-01 00:00:00")

averaged_field = {}

# Definir las estaciones del año
estaciones = {
                "Verano": [1, 2, 3], 
                "Fall": [4, 5, 6],
                "Invierno": [7, 8, 9],     
                "Primavera": [10, 11, 12]     
             }

# Iterar a través de las estaciones del año
for estacion, meses in estaciones.items():
    # Seleccionar los pasos de tiempo correspondientes a la estación actual
    # if estacion == "Primavera":
    pasos_tiempo_estacion = [t for t in time if t.month in meses]

    outfile = f"/media/javiera/Expansion/ATLAS_GAM/{dom}_{estacion}_{year}.dfsu"
    
    outfile_avg = f"/media/javiera/Expansion/ATLAS_GAM/{dom}_{estacion}_{year}_avg.dfsu"

    
    generic.extract(file,
                    outfile,
                    start=pasos_tiempo_estacion[0],
                    end=pasos_tiempo_estacion[-1],
                    )

    generic.avg_time(outfile,
                     outfile_avg)
    
#%%
import matplotlib.pyplot as plt
import mikeio 
import numpy as np
files = ("/media/javiera/Expansion/ATLAS_GAM/avg")
paths = list_files(files, ".dfsu") 
# dfs = mk.read(archivos[0])
# dfs.time.month

def cargar_y_plotear_dfus(paths):
    # Crear una figura con subplots 4x3
    fig, axs = plt.subplots(4, 3,
                            figsize=(16, 9),
                            dpi=300,
                            sharex=True,
                            sharey=True)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=-0.7)
    
    # Nombres de los meses
    season = ["Verano",
              "Invierno",
              "Fall",
              "Primavera",
             ]

    # Iterar sobre los archivos DFSU y crear subplots
    for i, path in enumerate(paths):
        # Cargar el archivo DFSU
        dfs = mikeio.read(path)
        
        # Obtener el tiempo
        tiempo = dfs.time
        
        
        # Obtener el año y mes del tiempo
        año = tiempo.year[0]
        mes = tiempo.month[0]
        
        # Encontrar la fila y columna en la que debe colocarse el subplot
        fila = (mes - 1) // 3 
        columna = año - 2016
        
        # Crear el subplot en la posición correspondiente
        ax = axs[fila, columna]
        
        # Plotear los datos
        dfs.AC_Age_concentration.plot.contourf(ax=ax,
                                      # title=None,
                                      label=None,
                                      figsize=(20,30),
                                     )
        # ax.plot(tiempo, datos_nodo[0], label=f"Nodo")
        ax.set_title(f"{mes}/{año}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        # ax.legend()
 
    # Ajustar la disposición de los subplots
    plt.tight_layout()
    
    # Mostrar el gráfico
    plt.show()

cargar_y_plotear_dfus(paths)
        
# %%
import pandas as pd

seasons = ["Verano",
          "Otoño",
          "Invierno",
          "Primavera",
         ]
dict_ds = {}
aux=[]
for season in seasons:
    list_ds = []
    
    for i, path in enumerate(paths):
        if season in path:
        # Cargar el archivo DFSU
            dfs = mikeio.read(path)
            list_ds.append(dfs)   
    dict_ds[season] = mk.Dataset.concat(list_ds)
    aux.append(dict_ds[season].mean(axis=0))
    
ds = mk.Dataset.concat(aux)
#%%
fig, axs = plt.subplots(2, 2,
                        figsize=(16, 9),
                        dpi=500,
                        sharex=True,
                        sharey=True)
fig.subplots_adjust(hspace=0.1,
                    wspace=-.8)
    
for i, ts in enumerate(ds.time):
    mes = ds.time[i].month
    
    # Encontrar la fila y columna en la que debe colocarse el subplot
    vec = [0,0,1,1]
    vec2 = [0,1,0,1]
    fila = vec[i] 
    columna = vec2[i]
    
    # Crear el subplot en la posición correspondiente
    ax = axs[fila, columna]
    
    # Plotear los datos
    ds.AC_Age_concentration.isel(time=i).sel(layers=20).plot(ax=ax,
                                  # title=None,
                                  label=None,
                                  figsize=(20,30),
                                 )
    ax.set_xlabel("lon")
    ax.set_title(seasons[i])
     
# Ajustar la disposición de los subplots
plt.tight_layout()

# Mostrar el gráfico
plt.show()
# ds.to_dfs("/home/javiera/Documents/IFOP/data/gam/gam_clim.dfsu")

    
#     # Obtener el tiempo
#     if path == paths[0]:
#         df = pd.DataFrame(dfs.AC_Age_concentration.geometry.element_coordinates,
#                               columns=["lon","lat","prof"])
#     tiempo = dfs.time
    
    
#     # Obtener el año y mes del tiempo
#     año = tiempo.year[0]
#     mes = tiempo.month[0]
    
    
#     df[f"{mes}_{año}"] = pd.Series(dfs.AC_Age_concentration.values.squeeze())

    