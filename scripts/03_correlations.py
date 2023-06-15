#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:38:30 2023

@author: mercedeszlehoczky
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

# Group the dataset by day and count the number of entries per day
daily_count = df_19.groupby('Date').size().reset_index(name='EntryCount')

#%% Columns of interest
columns_of_interest = ['AccidentType_en', 'AccidentSeverityCategory_en',
                       'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                       'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                       'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 
                       'Hr', 'RainDur', 'T', 'WVs', 'p']


#%% Create important variables df

df_select = df_19[['AccidentType_en', 'AccidentSeverityCategory_en',
                  'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                  'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                  'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 
                  'Hr', 'RainDur', 'T', 'WVs', 'p', 
                  'EntryCount']]



#%% ?????
    
# Group the dataset by day and count the number of entries per day
daily_count = df_19.groupby('Date').size().reset_index(name='EntryCount')


# Standardize the EntryCount variable
scaler = StandardScaler()
daily_count['EntryCount'] = scaler.fit_transform(daily_count[['EntryCount']])

# Merge the count of entries with the original dataset
merged_df = pd.merge(df_19, daily_count, on='Date')



# Compute the correlation matrix
corr_matrix = merged_df[['EntryCount', 'Variable1', 'Variable2', 'Variable3']].corr()




#%% Pearson correlation and plot
# using https://medium.com/the-researchers-guide/generate-numerical-correlation-and-nominal-association-plots-using-python-c8548aa4a663
    
# using dython library
from dython.nominal import associations
# Step 1: Instantiate a figure and axis object
fig, ax = plt.subplots(figsize=(16, 8))
# Step 2: Creating a pair-wise correlation plot 
# Saving it into a variable(r)
r = associations(df_select, ax = ax, cmap = "Blues")


#Pearson's
pearson_matrix = df_select.corr(method = 'pearson')

# Pearson's plot
# Step 1: Initiating a fig and axis object
fig, ax = plt.subplots(figsize=(18, 16))
# Step 2: Create a plot
cax = ax.imshow(pearson_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
# Step 3: Set axis tick labels
ax.set_xticks(ticks=range(len(pearson_matrix.columns)))
ax.set_xticklabels(pearson_matrix.columns, rotation=90)
ax.set_yticks(ticks=range(len(pearson_matrix.columns)))
ax.set_yticklabels(pearson_matrix.columns)
# Step 4: Resize the tick parameters
ax.tick_params(axis="both", labelsize=8)
# Step 5: Adding a color bar
fig.colorbar(cax).ax.tick_params(labelsize=8)
# Step 6: Add annotation
for (x, y), t in np.ndenumerate(pearson_matrix):
    ax.text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)

plt.show()


#%% Spearman's correlation for nonlinear associations
spearman_matrix = df_select.corr(method='spearman')

# Spearman's plot
# Step 1: Initiating a fig and axis object
fig, ax = plt.subplots(figsize=(18, 16))
# Step 2: Create a plot
cax = ax.imshow(spearman_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
# Step 3: Set axis tick labels
ax.set_xticks(ticks=range(len(spearman_matrix.columns)))
ax.set_xticklabels(spearman_matrix.columns, rotation=90)
ax.set_yticks(ticks=range(len(spearman_matrix.columns)))
ax.set_yticklabels(spearman_matrix.columns)
# Step 4: Resize the tick parameters
ax.tick_params(axis="both", labelsize=8)
# Step 5: Adding a color bar
fig.colorbar(cax).ax.tick_params(labelsize=8)
# Step 6: Add annotation
for (x, y), t in np.ndenumerate(spearman_matrix):
    ax.text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)

plt.show()

#%% Cramer’s V for categorical
#should select only categorical here

# Using association_metrics library
import association_metrics as am
# Convert columns to Category columns
selected_columns = ['AccidentType_en', 'AccidentSeverityCategory_en',
                  'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                  'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                  'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 
                  'EntryCount']
df_cramers = df_select[selected_columns].astype('category')
# Initialize a CramersV object using the pandas.DataFrame (df)
cramers_v = am.CramersV(df_cramers)
# It will return a pairwise matrix filled with Cramer's V, where 
# columns and index are the categorical variables of the passed     # pandas.DataFrame
cfit = cramers_v.fit().round(2)
cfit

# Instantiating a figure and axes object
fig, ax = plt.subplots(figsize = (18, 16))
# Generate a plot
cax = ax.imshow(cfit.values, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
# Step 3: Set axis tick labels
ax.set_xticks(ticks=range(len(cfit.columns)))
ax.set_xticklabels(cfit.columns, rotation=90)
ax.set_yticks(ticks=range(len(cfit.columns)))
ax.set_yticklabels(cfit.columns)
# Step 4: Resize the tick parameters
ax.tick_params(axis="both", labelsize=8)
# Adding a colorbar
fig.colorbar(cax).ax.tick_params(labelsize = 12)
# Adding annotations
for (x, y), t in np.ndenumerate(cfit):
    ax.annotate("{:.2f}".format(t),
                xy = (x, y),
                va = "center",
                ha = "center").set(color = "black", size = 8)



#%% Calculate Point-Biserial Correlation (binary-continuous)
# with https://towardsdatascience.com/point-biserial-correlation-with-python-f7cd591bd3b1

from scipy.stats import pointbiserialr

boolean_vars = ['Is_Holiday', 'Is_Weekend', 'Is_RushHour', 'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle']
pbc_matrix = []

for var in boolean_vars:
    r, p = pointbiserialr(df_select[var], df_select['EntryCount'])
    pbc_matrix.append((var, r, p))

pbc_matrix = pd.DataFrame(pbc_matrix, columns=['Variable', 'Correlation', 'p-value'])




#%% Calculate correlation between EntryCount and other variables

# Pearson 
# benchmarks linear relationship
from scipy.stats import pearsonr

correlations_pearson = {}
for col in columns_of_interest:
    col_values = []
    for date in daily_count['Date']:
        value = df_19.loc[df_19['Date'] == date, col].values[0]
        try:
            float_value = float(value)
            col_values.append(float_value)
        except (ValueError, TypeError):
            col_values.append(np.nan)
    
    corr, _ = spearmanr(daily_count['EntryCount'], col_values)
    correlations_pearson[col] = corr


# Spearman 
# benchmarks monotonic relationship
from scipy.stats import spearmanr

correlations_spearman = {}
for col in columns_of_interest:
    col_values = []
    for date in daily_count['Date']:
        col_values.append(df_19[df_19['Date'] == date][col].values[0])
    corr, _ = spearmanr(daily_count['EntryCount'], col_values)
    correlations_spearman[col] = corr

#See is the non-ordered correlations are the same ###DOUBLE CHECK THIS!!
matching_correlation = []

for key in correlations_pearson:
    if key in correlations_spearman:
        pearson_value = correlations_pearson[key]
        spearman_value = correlations_spearman[key]
        
        if pearson_value == spearman_value:
            matching_correlation.append((key, True))
        else:
            matching_correlation.append((key, False))
    else:
        matching_correlation.append((key, False))



    