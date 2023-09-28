#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:48:15 2023

@author: javiera

análisis exploratorio columna de agua en gam

"""
import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean 
from data_analysis import VariableInfo


plt.rcParams["font.family"] = ["Serif"]
plt.rcParams["font.size"] = 10

def month_to_season(month):
    if 1 <= month <= 3:
        return 'Summer'
    elif 4 <= month <= 5:
        return 'Autumn'
    elif 6 <= month <= 9:
        return 'Winter'
    else:
        return 'Spring'

def merge_ds(dom):
    var = ["od_mg",
            "nitra",
            "fos",
            "sil",
            "temp",
            "dens",
            "cl_tot",
            "sal",
            "nitri",
            "od_sat",
            "cl_act",
        ]

    ds_s = []
    for i in var:
        file = ("/home/javiera/Documents/IFOP/data/julia/"
                "fromjulia/"+dom+"_"+i+"_processed.nc")
        ds_aux = xr.open_dataset(file)
        ds_aux = ds_aux.where(ds_aux[i]>0)
        # ds_aux = ds_aux.sel(lon = ds_aux["lon"]>-.27)
        # ds_aux = ds_aux.sel(lat = ds_aux["lat"]>-52.3)
        ds_s.append(ds_aux)
        ds = xr.merge(ds_s)
    return ds


def get_coast():
    lcoast = ("/home/javiera/Documents/IFOP/data/"
              "coastline/CHILOE-MAGALLANES.shp")
    coast = gpd.read_file(lcoast)
    return coast

def scatter_plot(df, title, pathdir):
    fig, ax = plt.subplots(
                           dpi=300,
                           subplot_kw={'projection': 
                                       ccrs.PlateCarree()},
                           sharex=True,
                           sharey=True,)
    
    ax.set_extent([lonmin,
                   lonmax,
                   latmin,
                   latmax],
                       )
    
    coast = get_coast()
    
    coast.plot(ax=ax,
               facecolor="darkgray",
               edgecolors='k', lw=0.2)

    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels=True,
                      linewidth=0.6,
                      color='k',
                      alpha=0.3,
                      linestyle='--',
                      zorder=15)
    
    
    gl.top_labels = False
    gl.right_labels = False
    
    
    
    s=15
    
    sns.scatterplot(x='lon',
                    y='lat',
                    data=df,
                    # c="red",
                    # hue='ssn',
                    ax=ax,
                    ec="black",
                    # palette=paleta,
                    s=s,
                    alpha=.6,)
    plt.title(title)
    plt.savefig(pathdir, 
                bbox_inches="tight") 
    plt.show() 
    return fig, ax

def clim_scatter(df, start_month, end_month):
# def clim_scatter(ax, df, var, norm, start_month, end_month, ssn):
    
    df_aux = df.where((df.month >= start_month)
                      & (df.month <= end_month)
                      & (df.yr > 2020)
                      & (df.prof == 1)).dropna(how="all").reset_index(drop=True)
    return df_aux


path = ("/home/javiera/Documents/IFOP/data/")
path_figs = ("/home/javiera/Documents/IFOP/fig/python/gam/")

def set_extent(dom, ax):
    if dom == "gam":
        lonmin = -74.
        lonmax = -72.25
        latmin = -52.5
        latmax = -51.2
        ax.set_extent([lonmin,
                       lonmax,
                       latmin,
                       latmax],
                           )

        
# %% read data
dom = "gam"

df = pd.read_csv(path+dom+"/data_"+dom+"_sin_outliers.csv",
                 sep=" ")

df["station"] = df.groupby(["lon","lat"]).ngroup().astype("string")
df["season"] = df.month.map(month_to_season)
# df = df.where(df.yr > 2020).dropna(how="all")

variable_info = VariableInfo()

ds = merge_ds(dom)

# %% plot available data

lonmin = -74.
lonmax = -72.25
latmin = -52.5
latmax = -51.2

plot = "yes"
if plot == "yes":
    fig, axis = plt.subplots(ncols=3,
                             nrows=1,
                             figsize=(12, 8),
                             dpi=300,
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             sharex=True,
                             sharey=True,
                             )
    
    for ax,m in enumerate(df.season.unique()):
        df_aux = df.where(df.season == m).dropna(how="all")
        axis[ax].set_extent([lonmin,
                       lonmax,
                       latmin,
                       latmax],
                           )
        
        coast = get_coast()
        
        coast.plot(ax=axis[ax],
                   facecolor="darkgray",
                   edgecolors='k', lw=0.2)
    
        gl = axis[ax].gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          linewidth=0.6,
                          color='k',
                          alpha=0.3,
                          linestyle='--',
                          zorder=15)
        
        
        gl.top_labels = False
        gl.right_labels = False
        
        if ax != 0:
            gl.left_labels = False
        
        sns.scatterplot(x='lon',
                        y='lat',
                        data=df_aux,
                        # c="Reds",
                        # hue="yr",
                        ax=axis[ax],
                        ec="black",
                        # palette=paleta,
                        s=12,
                        alpha=1,
                        )
        axis[ax].set_title(f"{m}")




# %% PLOT DIVAnd fields

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
# ds = ds.where(ds[var]>=0)

for var in variable_info.info.keys():
    if var != "ac":

        fig, axis = plt.subplots(2,2, figsize=(16,10),
                               subplot_kw={'projection': ccrs.PlateCarree()},                          
                               dpi=300)
        
        vmin = variable_info.vmin(var)
        vmax = variable_info.vmax(var)
        dif = variable_info.dif(var)
        
        # Use appropriate start and end months based on time index
        
        
        for idx,mnth in enumerate(ds.time.dt.month.values):
        
            if mnth in range(4):
                start_month, end_month = 1, 3
            elif mnth  in range(4,7):
                start_month, end_month = 4, 6
            elif mnth in range(7,10):
                start_month, end_month = 7, 9
            elif mnth in range(10,12):
                start_month, end_month = 10, 12
                
            bounds = np.linspace(vmin,
                                 vmax,
                                 dif,
                                 endpoint=True)
            
            norm = cm.colors.BoundaryNorm(bounds,
                                          cmap[var].N)#, extend='both')
            
            sm = cm.ScalarMappable(norm=norm,
                                   cmap=cmap[var])
            # 
            row, col = divmod(idx,2)
            
            ax2 = axis[row, col]
            ax2.set_extent([lonmin,
                           lonmax,
                           latmin,
                           latmax],
                               )
            
            # cbar = ax2.figure.colorbar(sm)
            # cbar.set_label(var_info.label(var))
            
            ds[var].isel(depth=0, time=idx).plot(ax=ax2,
                                                x="lon",
                                                vmin=vmin,
                                                vmax=vmax,
                                                levels=bounds,
                                                cbar_kwargs={'ticks': bounds,
                                                              "label": variable_info.label(var)},
                                                    )
            
            df_aux = clim_scatter(df, start_month, end_month)
            sns.scatterplot(x='lon',
                            y='lat',
                            data=df_aux,
                            hue=var,
                            hue_norm=norm,
                            legend=False,
                            ax=ax2,
                            ec="black",
                            lw=2,
                            palette="viridis",
                            s=40,
                            alpha=1)
            
            gl = ax2.gridlines(crs=ccrs.PlateCarree(),
                              draw_labels=True,
                              linewidth=0.6,
                              color='k',
                              alpha=0.3,
                              linestyle='--',
                              zorder=15)
            
            
            gl.top_labels = False
            gl.right_labels = False
               
                
            ssn = month_to_season(mnth)
        
        
            ax2.set_title(ssn)
            coast = get_coast()
            coast.plot(ax=ax2,
                       facecolor="darkgray",
                       edgecolors="k",
                       lw=.2)
        
        
        
        plt.show()
        
        
        
        # fig.savefig(path_figs+f"scatter_{dom}_{var}_{ssn}.png",
        #             bbox_inches="tight")
        
        # Residual calculating and plotting
        fig, axis = plt.subplots(2,2, figsize=(16,10),
                                subplot_kw={'projection': ccrs.PlateCarree()},                          
                                dpi=300)
        
        
        
        
        for idx,mnth in enumerate(ds.time.dt.month.values):
         
            if mnth in range(4):
                start_month, end_month = 1, 3
            elif mnth  in range(4,7):
                start_month, end_month = 4, 6
            elif mnth in range(7,10):
                start_month, end_month = 7, 9
            elif mnth in range(10,12):
                start_month, end_month = 10, 12
        
            row, col = divmod(idx,2)
            
            ax2 = axis[row, col]
            ax2.set_extent([lonmin,
                           lonmax,
                           latmin,
                           latmax],
                               )
                  
            ds[f"{var}_relerr"].isel(depth=0, time=idx).plot(ax=ax2,
                                                      x="lon",
                                                      # vmin=vmin,
                                                      # vmax=vmax,
                                                      # levels=bounds,
                                                      cmap="coolwarm",
                                                      cbar_kwargs={#'ticks': bounds,
                                                                  "label": f"Error relativo {variable_info.title(var)}"},
                                                    )

            
            gl = ax2.gridlines(crs=ccrs.PlateCarree(),
                              draw_labels=True,
                              linewidth=0.6,
                              color='k',
                              alpha=0.3,
                              linestyle='--',
                              zorder=15)
            
            
            gl.top_labels = False
            gl.right_labels = False
               
                
            ssn = month_to_season(mnth)
        
            ax2.set_title(ssn)
            coast = get_coast()
            coast.plot(ax=ax2,
                       facecolor="darkgray",
                       edgecolors="k",
                       lw=.2)
    
# %% PYCNOCLINE SURFACES

# Define a threshold for density difference 

density_threshold_a = .1
density_threshold_b = .2

# Calculate density difference 
density_difference = abs(ds["dens_L1"].differentiate(coord="depth"))

# Find the mixed layer depth as the depth where the density difference

mixed_layer_depth = (ds['depth'].where(
                    (density_difference > density_threshold_a) &
                    (density_difference < density_threshold_b))
                    .min(dim='depth'))

# mixed_layer_depth.to_netcdf(f"{path}mixed_layer_depth.nc")

# Crear una figura y ejes con proyección cartográfica
fig, axis = plt.subplots(ncols=3,
                         nrows=1,
                         figsize=(12, 8),
                         dpi=300,
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         sharex=True,
                         sharey=True,
                         )
cbar_ax=fig.add_axes([0.4, 0.27, 0.2, 0.02])
# cbar_ax=fig.add_axes([0.92, 0.2, 0.02, 0.6])
vmax = 20
vmin = 0

# Iterar a través de las variables 'time' para trazar cada período de tiempo

for idx,mnth in enumerate(ds.time.dt.month.values):
 
    if mnth in range(4):
        start_month, end_month = 1, 3
    elif mnth  in range(4,7):
        start_month, end_month = 4, 6
    elif mnth in range(7,10):
        start_month, end_month = 7, 9
    elif mnth in range(10,12):
        start_month, end_month = 10, 12

    ax2 = axis[idx]
    # row, col = divmod(idx,2)
    # ax2 = axis[row, col]
    # ax2.set_extent([lonmin,
    #                lonmax,
    #                latmin,
    #                latmax],
    #                    )

    
    a=mixed_layer_depth.isel(time=idx).plot(x="lon",
                                        y="lat",
                                        cmap="gnuplot",
                                        ax=ax2,
                                        vmax=vmax,
                                        vmin=vmin,
                                        add_colorbar=False,
                                        )
    coast = get_coast()
    coast.plot(ax=ax2,
               facecolor="darkgray",
               edgecolors="k",
               lw=.2)
    set_extent(dom, ax2)
    ssn = month_to_season(ds.time.dt.month[idx])
    ax2.set_title(ssn)
    
cbar = fig.colorbar(a,
                    ax=axis.ravel().tolist(),
                    cax=cbar_ax,
                    orientation="horizontal",
                    # orientation="vertical",
                    # extend="both",  # Esto agrega flechas en los extremos de la barra de colores para indicar valores fuera de rango.
                    label="Mixed Layer Depth"
                    )

plt.tight_layout()
plt.show()

# %% read water age
import mikeio as mk
# calculate water age for the mixed layer depth
file = ("/home/javiera/Documents/IFOP/data/gam/"
        "gam_clim.dfsu")

dfs_or = mk.read(file)
dfs = dfs_or.copy()
dfs = dfs.isel(time=[0,1,3])


# %% Calculate the depth-integrated from surface to mix layer depth

variables_to_integrate = ["od_mg", 
                           "nitra",
                           "fos",
                           "sil",
                           "cl_tot",
                           "cl_act",
                           "nitri",
                           "sal"
                         ]  

integrated_values = {}

for variable in variables_to_integrate:
    # Select data from surface (depth=0) to mixing layer depth
    data_to_integrate = ds[f"{variable}"].where(ds['depth'] <= mixed_layer_depth)
    # Integrate along the depth dimension using the sum function
    integrated_values[variable] = data_to_integrate.mean(dim='depth')
    
    fig, axis = plt.subplots(1,3,
                             figsize=(8,8),
                             dpi=300,
                             subplot_kw=dict(
                                         projection=ccrs.PlateCarree()
                                         )
                             )
    
    for idx,mnth in enumerate(ds.time.dt.month.values):
     
        if mnth in range(4):
            start_month, end_month = 1, 3
        elif mnth  in range(4,7):
            start_month, end_month = 4, 6
        elif mnth in range(7,10):
            start_month, end_month = 7, 9
        elif mnth in range(10,12):
            start_month, end_month = 10, 12
    
        ax2 = axis[idx]



        a = integrated_values[variable].isel(time=idx).plot.contourf(ax=ax2,
                                       transform=ccrs.PlateCarree(),
                                       # cbar_kwargs=cbar_kwargs,
                                       # robust=False,
                                        # vmin=variable_info.vmin(variable),
                                        # vmax=variable_info.vmax(variable),
                                       add_colorbar=False)    
                                       
        
        ssn = month_to_season(integrated_values[variable].time[idx].dt.month)

    
        ax2.set_title(ssn)
    
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.05,
                        hspace=-0.3,
                        # left=0.1,
                        bottom=0,
                        # right=0.9,
                        # top=0.9,
                        )
    
    cbar_ax = fig.add_axes([0.2,
                            0.27,
                            0.6,
                            0.025
                            ])  # Left, bottom, width, height.
    
    cbar = fig.colorbar(a,
                        cax = cbar_ax,
                        # extend ='both',
                        orientation ='horizontal',
                        )
    
    cbar.set_label(variable_info.label(variable))

# %%  Combine the selected variables' data into a single DataFrame

df_data = pd.DataFrame([])
df_meta = (integrated_values["od_mg"]
               .to_dataframe()
                   .reset_index()[["lon",
                                   "lat",
                                   "time"]])

layer_aux = (mixed_layer_depth
                     .to_dataframe()
                         .reset_index()
                             .dropna()
                                 .reset_index(drop=True)
                                 
                                 
                             )

# Itera a través de las variables y agrega cada DataFrame resultante a df_boxplot
for variable in variables_to_integrate:
    df_variable = (integrated_values[variable]
                       .to_dataframe()
                       .reset_index())
    
    # Mantén solo la columna de la variable de interés
    df_variable = df_variable[[variable]]
    
    # Concatena el DataFrame de la variable con df_boxplot
    df_data[f"{variable}"] = df_variable
    

df_boxplot = pd.concat([df_meta,
                        df_data],
                       axis=1
                       )

df_boxplot.dropna(subset = variables_to_integrate,
                  how="all",
                  inplace=True,
                  )

df_boxplot.reset_index(inplace=True,
                       drop=True,
                       )

df_boxplot["depth"] = layer_aux["depth"]

df_boxplot.dropna(inplace=True
                  )

df_boxplot["season"] = df_boxplot.time.dt.month.map(month_to_season)
df_boxplot["ac"] = pd.Series([])
df_boxplot.reset_index(inplace=True,
                       drop=True,
                       )


# %% interpola valores de ac para los dataframe

for idx, prof in enumerate(df_boxplot.depth):
    depth = df_boxplot.depth[idx]
    if df_boxplot.season[idx] == "Summer":
        i_aux = 0
    elif df_boxplot.season[idx] == "Autumn":
        i_aux = 1
    elif df_boxplot.season[idx] == "Spring":
        i_aux = 2
    
    try:

        profile = dfs.sel(x=df_boxplot.lon[idx],
                y=df_boxplot.lat[idx]).isel(time=i_aux)
    
        df = pd.DataFrame(profile.geometry.element_coordinates,
                          columns=["lon", "lat", "z"]
                          )
    
        df["ac"] = profile.AC_Age_concentration.values
        
        df.sort_values(by="z",
                        ascending=False,
                        inplace=True)
    
        df.reset_index(drop=True, inplace=True)
        
        arg = np.argmax(df.z <= -depth)
        
        interp_val = df.iloc[:arg].mean()["ac"]
        
        df_boxplot.ac[idx] = interp_val
        
    except Exception as e:
            # Imprime el error, pero permite que el ciclo continúe con el siguiente
        print(f"Se produjo una excepción: {str(e)}.")

# df_boxplot.to_csv(f"{path}gam/integrated_mixed_layer_2.csv",
#                   sep=" ",
#                   index=False
#                   )

# %% v1
# initialize figure with 4 subplots in a row

(variables_to_integrate.append("ac")
 if "ac" not in variables_to_integrate else None)

plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(3, 3,
                       figsize=(10, 9),
                       dpi=300,
                       sharey=False,
                       sharex=True)

plt.subplots_adjust(wspace=0.5,
                    hspace=0.05,
                    # left=0.1,
                    bottom=0,
                    # right=0.9,
                    # top=0.9,
                    )
# draw boxplot for age in the 1st subplot
for idx,var in enumerate(variables_to_integrate):
    sns.boxenplot(data=df_boxplot, #vs boxplot
                   y=var,
                   x="season",
                   ax=ax.flatten()[idx],
                   # grid=False
                   ).set(ylabel=variable_info.label(var),
                         xlabel=None
                         )
    ax.flatten()[idx].grid(False)

plt.figure(dpi=300,
           figsize=(16, 6))

heatmap = sns.heatmap(df_boxplot[variables_to_integrate].corr(),
                      vmin=-1,
                      vmax=1, 
                    linewidths=.3,
                    annot=True,
                    fmt = ".2g",
                      cmap='viridis')

# %%

# Tratamiento de datos
# =============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Gráficos
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Configuración warnings
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

for ssn in df_boxplot.season.unique():
    data = df_boxplot.where(df_boxplot.season == ssn)[variables_to_integrate]
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
                           'PC8',
                           'PC9']
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
    print('Porcentaje de varianza explicada por cada componente')
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
    print('% de varianza explicada acumulada')
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
        columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', "PC9"],
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
    print('------------------')
    print('Valores originales')
    print('------------------')
    print(recostruccion.head())
    
    print('---------------------')
    print('Valores reconstruidos')
    print('---------------------')
    print(data.head())