#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:40:56 2023

@author: javiera

PLOT DE MAPAS

"""

# Importar librerias necesarias
import pandas as pd
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import random # para generar dataframe de ejemplo

# Define el tipo de fuente y tamaño
"""
Se puede definir al inicio del codigo para que sea uniforme en todos tus plot
"""
plt.rcParams["font.family"] = ["Serif"]
plt.rcParams["font.size"] = 12

# Funcion para cargar linea de costa
def get_coast():
    # lcoast = ("/tu/ruta/al/shapefile/"
    #           "/CHILOE-MAGALLANES.shp")
    lcoast = ("/home/javiera/Documents/IFOP/data/"
              "coastline/CHILOE-MAGALLANES.shp")
    coast = gpd.read_file(lcoast)
    return coast


# Setea los limites de tu mapa
lonmin = -74.
lonmax = -72.25
latmin = -52.5
latmax = -51.2

# Setea la proyeccion de tu mapa (usando libreria cartopy)
projection = ccrs.PlateCarree()

# Genera la figura
fig, ax = plt.subplots(
                       dpi=300, #a mayor dpi mayor resolucion de la figura, se recomienda 300
                       subplot_kw={'projection': 
                                   projection},
                       sharex=True, #si quieres que tus subplots compartan ejes
                       sharey=True,
                      )

ax.set_extent([lonmin,
               lonmax,
               latmin,
               latmax],
                   )

coast = get_coast()

coast.plot(ax=ax,
           facecolor="darkgray", # Color del continente
           edgecolors='k', # Color de la linea de costa
           lw=0.2, # Linewidth linea de costa
           )

# Agrega grillas en tu mapa
gl = ax.gridlines(crs=projection,
                  draw_labels=True,
                  linewidth=0.6,
                  color='k',
                  alpha=0.3,
                  linestyle='--',
                  zorder=15)

# Quita label en bordes
gl.top_labels = False
gl.right_labels = False


# Finalmente añade tu data
"""
En este ejemplo se añade un scatter plot con la libreria seaborn,
los datos corresponden a un DataFrame (df) hecho en Pandas,
aca debes añadir la informacion que quieras visualizar, recordando plotear 
sobre el mismo eje definido en la linea 49 (ax) 
"""

df = pd.DataFrame({"lon": [random.uniform(lonmin, lonmax) for _ in range(20)],
                   "lat": [random.uniform(latmin, latmax) for _ in range(20)],
                   "data": [random.uniform(10, 35) for _ in range(20)],
                   }
                  )
s=30

sns.scatterplot(x='lon',
                y='lat',
                data=df,
                # c="red",
                hue='data',
                ax=ax, #la nueva figura se superpone en el eje definido al 
                       # inicio, (se puede imaginar como una construccion de capas para crear la figura)
                ec="black",
                # palette=paleta,
                s=s,
                alpha=.6,)

plt.title("Tu titulo")

## Descomenta estas lineas para guardar tu figura

# plt.savefig("Path/donde/guardaras/lafigura", 
#             bbox_inches="tight") 

plt.show() 