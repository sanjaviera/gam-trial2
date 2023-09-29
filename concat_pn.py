#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:24:13 2023

@author: javiera
"lee datos de matriz historica"
"""

import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_analysis import VariableInfo, visualize_outliers, grafica_perfil
import scipy

variable_info = VariableInfo()
plt.rcParams["font.family"] = ["Serif"]
plt.rcParams["font.size"] = 12

def rename_df(data):
    df = data.rename(columns={"Latitude": "lat",
                              "Longitude": "lon",
                              "Day": "day",
                              "Month": "month",
                              "Year": "yr",
                              "Hour": "hr",
                              "Hour (UTC)": "hr",
                              "Minute": "min",
                              "Second": "seg",
                              "Cruise":"cruise",
                              "Station": "stat",
                              "Profundidad (m)": "prof",
                              "Salinidad (psu)": "sal",
                              "Salinidad (PSS-78)":"sal",
                              "Temperatura (°C)": "temp",
                              "Sigma-t": "dens",
                              "Fluorescencia (µg/L)" : "fl",
                              # "Fluorescencia (mg/m3)": "fl",
                              "oxigeno dis. (µM)": "od_mol",
                              "Oxigeno dis. (ml/L)": "od_ml",
                              "Oxigeno dis. (mg/L)": "od_mg",
                              "oxigeno sat. (%)": "od_sat",
                              "oxigeno dis. (µmol/L)": "od_mol_l",

                              "oxigeno dis, (µM)": "od_mol",
                              "Oxigeno dis, (ml/L)": "od_ml",
                              "Oxigeno dis, (mg/L)": "od_mg",
                              "oxigeno sat, (%)": "od_sat",
                              "oxigeno dis, (µmol/L)": "od_mol_l",

                              "conductividad (S/m)": "cond",
                              "taza descenso (m/s)": "taza_descenso",
                              "nitrato (µM)": "nitra",
                              "nitrito (µM)": "nitri",
                              "fosfato (µM)": "fos",
                              "silicato (µM)": "sil",
                              "Clorofila total (µg/L)": "cl_tot",
                              "Clorofila activa (µg/L)": "cl_act",
                              "Feopigmentos (µg/L)": "feop",
                            # "nitrato (µmol/L)": "nitra",
                            # "nitrito (µmol/L)": "nitri",
                            # "fosfato (µmol/L)": "fos",
                            # "silicato (µmol/L)": "sil",
                            },
                    )
    return df

def process_df(file_path, outliers=False):
    try:
        # Intentar leer todas las hojas del archivo Excel
        try:
            sheet_names = pd.ExcelFile(file_path).sheet_names
        except:
            sheet_names = None

        # Si hay múltiples hojas, procesar cada hoja
        if sheet_names:
            dfs = []
            for sheet_name in sheet_names:
                data = pd.read_excel(file_path,
                                     skiprows=lambda x: x in range(8),
                                     header=[1],
                                     sheet_name=sheet_name)
                df = rename_df(data)
                dfs.append(df)

            concatenated_df = pd.concat(dfs, ignore_index=True)
        else:
            # Si no hay hojas, leer solo el contenido del archivo
            data = pd.read_excel(file_path,
                                            skiprows=lambda x: x in range(8),
                                            header=[1])
            concatenated_df = rename_df(data)

        concatenated_df = concatenated_df.drop_duplicates()
        concatenated_df = concatenated_df.replace(-9999.0, np.nan)    
        concatenated_df = concatenated_df.replace("NaN", np.nan)    
        concatenated_df = concatenated_df.replace("NAN", np.nan)    
        
        if outliers:
            concatenated_df = visualize_outliers(concatenated_df)
        
        return concatenated_df
    except Exception as e:
        print("Error:", e)
        return None

    
# %%

if __name__ == '__main__':
    
    fn = ("/home/javiera/Documents/IFOP/data/"
          "datospatagonia_updated/"
          "1_Patagonia_norte_2019-2020_(ETAPA_1).xlsx")

    df = process_df(fn)
    
    print(df)

    umbral = 10 # std_threshold
    visualize_outliers(df, umbral)
    
    grafica_perfil(df, 43858, "sil") # eliminar
    grafica_perfil(df, 44071, "cl_tot") # eliminar
    
    # z_score, z_score_abs = visualize_outliers(df)
    # np.where((z_score_abs > std_threshold)["sal"] == True)


# df.to_csv(("/home/javiera/Documentos/ifop/Escalas/"
#            "2022-2023/data/chiloe_aysen_2010-2021/matriz_patagonianorte.csv"),
#           sep=" ",
#           index=False)
