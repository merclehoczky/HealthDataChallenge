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

#%% Indicate holidays 
# from https://www.zh.ch/content/dam/zhweb/bilder-dokumente/footer/arbeiten-fuer-den-kanton/personalamt/Feiertage2019.pdf

holiday_dates = ['2019-01-01', '2019-01-02',
                 '2019-04-08', '2019-04-18', '2019-04-19', '2019-04-22',
                 '2019-05-01', '2019-05-29', '2019-05-30',
                 '2019-06-10', 
                 '2019-08-01', 
                 '2019-09-09',
                 '2019-12-24',  '2019-12-25',  '2019-12-26',  '2019-12-31']
                 

df_19['Is_Holiday'] = np.nan

df_19['Date'] = pd.to_datetime(df_19['Date'])

# Set 'Is_Holiday' column based on holiday dates
df_19['Is_Holiday'] = df_19['Date'].isin(holiday_dates)

   
#%% Indicate weekends

weekends = ['Saturday', 'Sunday']

df_19['Is_Weekend'] = np.nan


df_19['Is_Weekend'] = df_19['AccidentWeekDay_en'].isin(weekends)


#%% Going out

df_19['Is_GoingOut'] = np.nan

#ok
df_19['Is_GoingOut'] = np.where((df_19['AccidentWeekDay_en'].isin(weekends)) | 
                                (df_19['AccidentWeekDay_en'] == 'Friday') | 
                                (df_19['Date'].isin(holiday_dates)) |
                                (df_19['Date'] - pd.DateOffset(days=1)).isin(holiday_dates), 'True', 'False')


#better
df_19['Is_GoingOut'] = np.where((df_19['AccidentWeekDay_en'].isin(weekends)) | 
                                ((df_19['AccidentWeekDay_en'] == 'Friday') & (~df_19['Date'].isin(holiday_dates))) |
                                (df_19['Date'].isin(holiday_dates)), 'True', 'False')
#####USE SHIFT

#%% Rush hour
# from https://www.tomtom.com/traffic-index/zurich-traffic/

df_19['Is_RushHour'] = np.nan
rushhour = [7,8,16,17,18]

df_19['Is_RushHour'] = df_19['AccidentHour'].isin(rushhour)


#%% Summary statistics
sumstat= df_19.describe()



# %% Exploration


#%% Plots
plt.hist(df_19['Date'], bins = 365)

# Plot accident type per month 
sns.histplot(x='AccidentType_en', data=df_19, kde=False, hue='AccidentMonth')
plt.xticks(rotation = 45) 
plt.show()


# Plot air pressure
sns.histplot(x='p', data=df_19, kde=False, hue='AccidentMonth')

# Plot accident severity histogram
sns.histplot(x='AccidentSeverityCategory_en', data=df_19, kde=False)
plt.xticks(rotation = 45) 
plt.show()

# Plot road type histogram
sns.histplot(x='RoadType_en', data=df_19, kde=False)
plt.xticks(rotation = 45) 
plt.show()

# Plot AccidentMonth_en histogram
sns.histplot(x='AccidentMonth_en', data=df_19, kde=False)
plt.xticks(rotation = 45) 
plt.show()

# Plot AccidentWeekDay_en histogram
sns.histplot(x='AccidentWeekDay_en', data=df_19, kde=False)
plt.xticks(rotation = 45) 
plt.show()

# Plot AccidentHour_text histogram
sns.histplot(x='AccidentHour_text', data=df_19, kde=False)
plt.xticks(rotation = 45) 
plt.show()

# Plot AccidentType_en vs RainDur
plt.plot(df_19.AccidentType_en, df_19.RainDur, 'o')

# Plot AccidentType_en vs T
plt.plot(df_19.AccidentInvolvingPedestrian, df_19.EntryCount, 'o')

# Plot AccidentType_en vs RoadType_en
plt.plot(df_19.AccidentType_en, df_19.RoadType_en, 'o')


# Plot AccidentType_en vs AccidentSeverityCategory_en

