#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:14:37 2023

@author: javiera

isopycnal surfaces using divand

"""

import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

class VariableInfo:
    def __init__(self):
        self.info = {
            'ac': {'vmin': 10,
                   'vmax': 930,
                   'dif': len(np.linspace(10, 930, 10).astype("int")),
                   "label": "Edad del agua [días]",
                   "title": "Edad del agua"
                    },
          
            'temp': {'vmin': 5,
                     'vmax': 13,
                     'dif': 11,
                     "label": "Temperatura [ºC]",
                     "title": "Temperatura"
                      },
            
            'sal': {'vmin': 2,
                       'vmax': 24,
                       'dif': 13,
                       "label": "Salinidad [PSU]",
                       "title": "Salinidad"
                      },
            
            # 'fl': {'vmin': 0,
            #            'vmax': 10,
            #            'dif': 11,
            #            "label": "Fluorescencia (µg/L)",
            #            "title": "Fluorescencia"
            #           },
            
            'cl_tot': {'vmin': 0,
                       'vmax': 9,
                       'dif': 10,
                       "label": "Clorofila total [µg/L]",
                       "title": "Clorofila total"
                      },
            
            'cl_act': {'vmin': 0,
                       'vmax': 9,
                       'dif': 10,
                       "label": "Clorofila activa [µg/L]",
                       "title": "Clorofila activa"
                      },
            
            'sil': {'vmin': 0,
                    'vmax': 15,
                    'dif': 16,
                    "label": "Silicato [µM]",
                    "title": "Silicato"
                   },
            
            'fos': {'vmin': 0,
                    'vmax': 2,
                    'dif': 5,
                    "label": "Fosfato [µM]",
                    "title": "Fosfato"
                    },
            
            'nitra': {'vmin': 0,
                      'vmax': 15,
                      'dif': 16,
                      "label": "Nitrato [µM]",
                      "title": "Nitrato"
                      },
            
            'nitri': {'vmin': 0,
                      'vmax': 0.55,
                      'dif': 10,
                      "label": "Nitrito [µM]",
                      "title": "Nitrito"
                      },
            
            'od_mg': {'vmin': 9,
                      'vmax': 13,
                      'dif': 8,
                      "label": "Oxígeno disuelto [mg/L]",
                      "title": "Oxígeno disuelto"
                      },
            
            "dens": {"vmin":2,
                     "vmax":30,
                     "dif":15,
                     "label": "Densidad [Sigma-t]",
                     "title": "Densidad"
                     }
        }

    def __getitem__(self, var):
        return self.info[var]

    def vmin(self, var):
        return self.info[var]['vmin']

    def vmax(self, var):
        return self.info[var]['vmax']

    def dif(self, var):
        return self.info[var]['dif']

    def label(self, var):
        return self.info[var]['label']

    def title(self, var):
        return self.info[var]['title']


def merge_ds(dom):
    var = ["od_mg",
           "nitra",
           "fos",
           "sil",
           "temp",
           "dens",
           "cl_tot",
           "sal"
           ]

    ds_s = []
    for i in var:
        file = ("/home/javiera/Documents/IFOP/data/julia/"
                "fromjulia/"+dom+"_"+i+"_processed.nc")
        ds_aux = xr.open_dataset(file)
        # ds_aux = ds_aux.where(ds_aux[i]>0)
        ds_s.append(ds_aux)
        ds = xr.merge(ds_s)
    return ds
    
def grafica_perfil(df, idx, var):
    subset = df[(df["lon"] == df.lon.iloc[idx]) &
                (df["lat"] == df.lat.iloc[idx])]
    
    subset[[var,
            "prof"]].dropna().plot(x=var,
                                   y="prof",
                                   marker="o",
                                   figsize=(8, 6))
                                   
    plt.gca().invert_yaxis()
    return subset

def visualize_outliers(df, umbral):
    describe_df_aux = df.describe()
    describe_num_df = describe_df_aux.reset_index()

    describe_df = describe_num_df[describe_num_df["index"] != "count"]
    num_col = df._get_numeric_data().columns

    for i in num_col:
        if i in ["index"]:
            continue
        fig, ax = plt.subplots(nrows=2, ncols=1, dpi=300)
        sns.stripplot(x="index", y=i, data=describe_df, ax=ax[1])
        sns.boxplot(x=i, data=df, ax=ax[0])
        plt.suptitle(i)
        plt.show()

    
    z_scores = scipy.stats.zscore(df[num_col[9:]],
                                  nan_policy="omit")
    z_scores_abs = abs(z_scores)
    
    z_scores_show = z_scores_abs.apply(pd.Series.max)
    idx = z_scores_abs.apply(pd.Series.argmax)
    print(z_scores_show)
    print(idx)
    
    std_threshold = umbral
    num_values_above_threshold = (z_scores_abs > std_threshold).sum(axis=0)
    print(f"cantidad que supera el umbral de {std_threshold} stds")
    print(num_values_above_threshold)
    id_aux = num_col.drop(['lat',
                           'lon',
                           'day',
                           'month', 
                           'yr',
                           'hr',
                           'min', 
                           'seg',
                           'prof'])
    df[id_aux] = df[id_aux][z_scores_abs < std_threshold]
    return z_scores, z_scores_abs, df
        
#%%

if __name__ == '__main__':
    
    dom = "gam"
    ds = merge_ds(dom)    


    # Define a threshold for density difference (adjust this value based
    # on your dataset)
    density_threshold_a = 0.09
    density_threshold_b = 0.15
    
    # Calculate density difference 
    density_difference = ds["dens"].differentiate(coord="depth")
    
    # Find the mixed layer depth as the depth where the density difference
    # is below the threshold
    mixed_layer_depth = (ds['depth'].where(
                        (density_difference > density_threshold_a) &
                        (density_difference < density_threshold_b))
                        .min(dim='depth')).astype(int)
    
    # Calculate the depth-integrated values for each variable
    variables_to_integrate = ['nitra']  # Add more variables as needed
    
    integrated_values = {}
    
    for variable in variables_to_integrate:
        # Select data from surface (depth=0) to mixing layer depth
        data_to_integrate = ds[variable].where(ds['depth'] <= mixed_layer_depth)
        # Integrate along the depth dimension using the sum function
        integrated_values[variable] = data_to_integrate.mean(dim='depth')
    
    
    
    # # Pretty much seems to work.
