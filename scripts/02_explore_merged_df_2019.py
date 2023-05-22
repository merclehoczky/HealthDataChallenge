#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:36:40 2023

@author: mercedeszlehoczky
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# %% Merge datasets
df_19 = dd.merge(accidents, w19, how='inner', on=['Datum'])

#%% Drop NaNs, nonsense values
df_19.dtypes
#Fix datatypes
df_19['AccidentType'] = df_19['AccidentType'].astype('category')
df_19['AccidentType_en'] = df_19['AccidentType_en'].astype('category')
df_19['AccidentSeverityCategory'] = df_19['AccidentSeverityCategory'].astype('category')
df_19['AccidentSeverityCategory_en'] = df_19['AccidentSeverityCategory_en'].astype('category')
df_19['RoadType'] = df_19['RoadType'].astype('category')
df_19['RoadType_en'] = df_19['RoadType_en'].astype('category')
df_19['AccidentYear'] = df_19['AccidentYear'].astype('category')
df_19['AccidentMonth'] = df_19['AccidentMonth'].astype('category')
df_19['AccidentMonth_en'] = df_19['AccidentMonth_en'].astype('category')
df_19['AccidentWeekDay'] = df_19['AccidentWeekDay'].astype('category')
df_19['AccidentWeekDay_en'] = df_19['AccidentWeekDay_en'].astype('category')
df_19['AccidentHour'] = df_19['AccidentHour'].astype('category')
df_19['AccidentHour_text'] = df_19['AccidentHour_text'].astype('category')

# Drop NaNs
df_19 = df_19.dropna()

# Summary statistics
sumstat= df_19.describe()



# %% Exploration


#%% Plots
plt.hist(df_19['Date'], bins = 365)

sns.histplot(x='AccidentType_en', data=df_19, kde=False, hue='AccidentMonth')
plt.xticks(rotation = 45) 
plt.show()
#%% Correlations