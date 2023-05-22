#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:51:17 2023

@author: mercedeszlehoczky
"""

import pandas as pd
import numpy as np

# %% Import dataset
w19_hr = pd.read_csv('../data/2019/ugz_ogd_meteo_h1_2019.csv', header = 0)
w19_hr

# %% View data
w19_hr.dtypes

print(w19_hr['Standort'].unique())

print(w19_hr['Parameter'].unique())

# %% Filter parameters
conditions = ['StrGlo', 'WD', 'WVv']
filtered = w19_hr.loc[~w19_hr['Parameter'].isin(conditions)]
filtered.head()

# %% # Create new dataframe with locations sorted under timepoints (long to wide)

w19_new = pd.pivot(filtered, index = ['Datum', 'Standort'], 
                   columns = ['Parameter'], values = ['Wert'])

# View levels of indices on columns
w19_new.columns.levels
w19_new.columns.get_level_values(1)
w19_new.columns = w19_new.columns.droplevel() # Drop outermost level (Wert)

w19_new.head(50)

# %% Averaged values under each timepoint

# Save dates 
dates = w19_new.reset_index(inplace=False) #Save dates as new column (were indices here)
dates = dates['Datum']                   # Name the column
dates = dates.drop_duplicates()          # Drop multiplicates (each day was x5 or x8)
dates = dates.to_frame()                 # Turn list into df
dates = dates.reset_index(drop = True)   # Reindexing
dates

# Average values
w19_avg = w19_new.groupby(np.arange(len(w19_new))//3).mean()

# Join average with dates
    # Make sure that indices are the same
w19_avg = w19_avg.reset_index(drop=True)
w19_avg.index

dates = dates.reset_index(drop=True)
dates.index

#Merge
w19 = pd.merge(w19_avg, dates, left_index = True, right_index = True)

# Rearrange columns
w19 = w19[['Datum', 'Hr', 'RainDur', 'T', 'WVs', 'p']]

w19.head(50)
