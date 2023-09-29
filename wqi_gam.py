#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:57:03 2023

@author: javiera
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import statsmodels.api as sm
from data_analysis import VariableInfo
import matplotlib.cm as cm
import cmocean

import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = ["Serif"]
plt.rcParams["font.size"] = 10

def get_coast():
    lcoast = ("/home/javiera/Documents/IFOP/data/"
              "coastline/CHILOE-MAGALLANES.shp")
    coast = gpd.read_file(lcoast)
    return coast

path = ('/home/javiera/Documents/IFOP/data/')

df = pd.read_csv(f"{path}gam/integrated_mixed_layer.csv",
                 sep=" ")


variables_numeric = ["od_mg", 
                      "nitra",
                      "fos",
                      "sil",
                      "cl_tot",
                      # "cl_act",
                      "nitri",
                      # "sal",
                      "ac"
                    ] 

# %% datos
lonmin = -74.
lonmax = -72.25
latmin = -52.5
latmax = -51.2
s=7

cmap = {'od_mg': cmocean.cm.ice,
        'temp': cmocean.cm.ice,
        'sal': cmocean.cm.ice,
        'nitra': cmocean.cm.ice,
        'nitri': cmocean.cm.ice,
        'fos': cmocean.cm.ice,
        'sil': cmocean.cm.ice,
        'feop': cmocean.cm.ice,
        'cl_tot': cmocean.cm.ice,
        'cl_act': cmocean.cm.ice,
        'dens': cmocean.cm.ice,
       }

for ssn in df.season.unique():
    
    data = df.where(df.season == ssn).dropna(how="all")
    
    fig, ax = plt.subplots(nrows=3,
                           ncols=3,                       
                           dpi=300,
                           subplot_kw={'projection': 
                                       ccrs.PlateCarree()},
                           sharex=True,
                           sharey=True,
                           )
    fig.suptitle(f"{ssn}")
    
    
    for var, axis in enumerate(ax.flatten()):
        axis.set_extent([lonmin,
                       lonmax,
                       latmin,
                       latmax],
                       )
        # variable_info = VariableInfo()
        # vmin = variable_info.vmin(var)
        # vmax = variable_info.vmax(var)
        # dif = variable_info.dif(var)
        
        # bounds = np.linspace(vmin,
        #                      vmax,
        #                      dif,
        #                      endpoint=True)
        
        # norm = cm.colors.BoundaryNorm(bounds,
        #                               cmap[var].N)#, extend='both')
        
        # sm = cm.ScalarMappable(norm=norm,
        #                        cmap=cmap[var])
        
        sns.scatterplot(x='lon',
                        y='lat',
                        data=data,
                        # c="red",
                        hue=variables_numeric[var],
                        # hue_norm=norm,
                        ax=axis,
                        ec=None,
                        legend=False,
                        # palette=paleta,
                        s=s,
                        alpha=.6,
                        )
        
        coast = get_coast()
        
        coast.plot(ax=axis,
                   facecolor="darkgray",
                   edgecolors='k', lw=0.2)
        
        gl = axis.gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          linewidth=0.6,
                          color='k',
                          alpha=0.3,
                          linestyle='--',
                          zorder=15)
        
        
        gl.top_labels = False
        gl.right_labels = False
        

    # plt.title(title)
    # plt.savefig(pathdir, 
    #             bbox_inches="tight") 
    # plt.show() 
    
aux = df.groupby("season").describe().T

# %% version nueva

    
data = df[variables_numeric].dropna()

print('----------------------')
print(f"Media de cada variable {ssn}")
print('----------------------')
print(data.mean(axis=0))

print('-------------------------')
print(f'Varianza de cada variable {ssn}')
print('-------------------------')
print(data.var(axis=0))
data.dropna(how="any", inplace=True)

# Entrenamiento modelo PCA con escalado de los datos
# ==============================================================================
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(data)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

# dataframe de los loadings
df_pca = pd.DataFrame(
            data    = modelo_pca.components_,
            columns = data.columns,
            index   = ['PC1',
                       'PC2',
                       'PC3',
                       'PC4',
                       'PC5',
                       'PC6',
                       'PC7',
                       # 'PC8',
                       # 'PC9']
                       ]
            )

# % Heatmap componentes
# ==============================================================================
plt.rcParams["font.size"] = 5
fig, ax = plt.subplots(dpi=300,
                       nrows=1,
                       ncols=1,
                       figsize=(4, 2)
                       )

componentes = modelo_pca.components_
# plt.imshow(componentes.T,
#            cmap='viridis',
#            aspect='auto'
#            )

sns.heatmap(componentes.T,
            cmap="BrBG",
            # cbar_kws={'fontsize': 5,},
            linewidths=.3,
            annot=True,
            fmt = ".2g",
            ax=ax,
            annot_kws={'size': 5,},
            vmin=-1,
            vmax=1,
            )


plt.yticks(range(len(data.columns)),
           data.columns,
           fontsize=5,
           rotation=0,
           )

plt.xticks(range(len(data.columns)),
           np.arange(modelo_pca.n_components_) + 1,
           fontsize=5,
           )

ax.set_xlabel("Componentes",
              fontsize=5,)

plt.grid(False)
# plt.colorbar();

# % Porcentaje de varianza explicada por cada componente
# ==============================================================================
print('----------------------------------------------------')
print(f'Porcentaje de varianza explicada por cada componente {ssn}')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(dpi=300,
                       nrows=1,
                       ncols=1,
                       figsize=(6, 4)
                       )
ax.bar(x = np.arange(modelo_pca.n_components_) + 1,
       height = modelo_pca.explained_variance_ratio_
       )

for x, y in zip(np.arange(len(data.columns)) + 1,
                modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
                )       

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title(f'Porcentaje de varianza explicada por cada componente {ssn}')
ax.set_xlabel('Componente principal')
ax.set_ylabel('% varianza explicada');

# Porcentaje de varianza explicada acumulada
# ==============================================================================
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print(f'% de varianza explicada acumulada {ssn}')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(dpi=300,
                       nrows=1,
                       ncols=1,
                       figsize=(6, 4)
                       )
ax.plot(
        np.arange(len(data.columns)) + 1,
        prop_varianza_acum,
        marker = 'o'
        )

for x, y in zip(np.arange(len(data.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
                )
    
ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title(f'Porcentaje de varianza explicada acumulada {ssn}')
ax.set_xlabel('Componente principal')
ax.set_ylabel('% varianza acumulada');

# Proyección de las observaciones de entrenamiento
# ==============================================================================
proyecciones = pca_pipe.transform(X=data)
proyecciones = pd.DataFrame(
    proyecciones,
    columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', ],
               # 'PC8', "PC9"],
    index   = data.index
)
proyecciones.head()

# Recostruccion de las proyecciones
# ==============================================================================
recostruccion = pca_pipe.inverse_transform(proyecciones)
recostruccion = pd.DataFrame(
                    recostruccion,
                    columns = data.columns,
                    index   = data.index
)
# print('------------------')
# print('Valores originales')
# print('------------------')
# print(recostruccion.head())

# print('---------------------')
# print('Valores reconstruidos')
# print('---------------------')
# print(data.head())
