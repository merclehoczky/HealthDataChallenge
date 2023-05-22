#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:02:10 2023

@author: mercedeszlehoczky
"""

import pandas as pd
import numpy as np
import datetime
import calendar

# Enlarge output display capacity
pd.set_option('display.max_rows', 500)

# Import Zuerich city accidents dataset
accidents = pd.read_csv('../data/RoadTrafficAccidentLocations.csv', header = 0)

# View an instance
accidents.loc[34649, :]

# Drop duplicate columns, only keep English ones
accidents = accidents.drop(columns = ['AccidentType_de', 'AccidentType_fr', 'AccidentType_it', 
                      'AccidentSeverityCategory_de', 'AccidentSeverityCategory_fr', 'AccidentSeverityCategory_it',
                      'RoadType_de', 'RoadType_fr', 'RoadType_it',
                      'AccidentMonth_de', 'AccidentMonth_fr', 'AccidentMonth_it',
                      'AccidentWeekDay_de', 'AccidentWeekDay_fr', 'AccidentWeekDay_it'])

#%% Quick view

# View datatypes 
accidents.dtypes

# View dates 
print(accidents['AccidentWeekDay'].unique())

print(accidents['AccidentHour'].unique())

print(accidents['AccidentHour_text'].unique())

accidents[['AccidentYear', 'AccidentMonth_en','AccidentWeekDay', 'AccidentWeekDay_en','AccidentHour_text']].head(20)

accidents[['AccidentYear', 'AccidentMonth_en','AccidentWeekDay', 'AccidentWeekDay_en','AccidentHour_text']].tail(20)

# Dataset is until 2022 december
#%% Preprocessing

# Only keep data in 2019

accidents = accidents[accidents.AccidentYear == 2019]

#%%% Create date fixing loop
# Create new column for date
accidents["DayCount"] = np.nan

# Reset indices             
accidents = accidents.reset_index(drop=True)           

# Initialise counter
count = 1    

# Loop though data and add day counter
for row in accidents.index:
    accidents.at[row, 'DayCount'] = count   #Set first day as 1
    if (accidents.at[row, 'AccidentWeekDay'] != accidents.at[row+1, 'AccidentWeekDay']): # If the days change
        count = count + 1 # Add to the counter (change the dates)
        accidents.at[row+1, 'DayCount'] = count  #Add counter as day counter
         

accidents['DayCount'] = accidents['DayCount'].astype(int)

accidents['Date'] = np.nan

def day_to_date(year, day_number):
    if calendar.isleap(year):
        days_in_year = 366
    else:
        days_in_year = 365
    if day_number < 1 or day_number > days_in_year:
        return None
    date = datetime.date.fromordinal(datetime.date(year, 1, 1).toordinal() + day_number - 1)
    return date

for row in accidents.index:
    year = accidents.at[row, 'AccidentYear']
    day_number = accidents.at[row, 'DayCount']
    accidents['Date'] = accidents.apply(lambda row: day_to_date(row['AccidentYear'], row['DayCount']), axis=1)


# Here fix date format to YYYY-MM-ddThh:00+01:00 OR break that up in the weather/air quality dataset


# Assuming your DataFrame is named "df" and the date column is named "date" and the hour column is named "hour"

# Combine the date and hour columns into a new column
accidents['Datum'] = pd.to_datetime(accidents['Date']) + pd.to_timedelta(accidents['AccidentHour'], unit='h')
accidents['Datum'] = accidents['Datum'].dt.strftime('%Y-%m-%dT%H:00+0100')






#%% Preprocessing and descriptive statistics
summary = accidents.describe(datetime_is_numeric = True)

print(accidents['AccidentType_en'].value_counts())

print(accidents['AccidentSeverityCategory_en'].value_counts())

print(accidents['AccidentHour'].value_counts())


