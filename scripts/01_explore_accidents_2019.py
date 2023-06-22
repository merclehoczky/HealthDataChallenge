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
import matplotlib.pyplot as plt

# Enlarge output display capacity
pd.set_option('display.max_rows', 500)

# Import Zuerich city accidents dataset
accidents = pd.read_csv('data/RoadTrafficAccidentLocations.csv', header = 0)

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

#%%% Create date fixing loops

# Create new column for day count
accidents["DayCount"] = np.nan

# Reset indices             
accidents = accidents.reset_index(drop=True)           

# Initialise counter
count = 1    


# Loop though data and add day counter
for row in range(len(accidents.index)):
    accidents.at[row, 'DayCount'] = count   #Set first day as 1
    if row < len(accidents.index) - 1 and (accidents.at[row, 'AccidentWeekDay'] != accidents.at[row+1, 'AccidentWeekDay']): # If the days change
        count = count + 1 # Add to the counter (change the dates)
        accidents.at[row+1, 'DayCount'] = count  #Add counter as day counter
         

# Convert it to int
accidents['DayCount'] = accidents['DayCount'].astype(int)



# Add dates with uniform format YYYY-MM-dd

# Create new column
accidents['Date'] = np.nan

# Define function for conversion
def day_to_date(year, day_number):
    if calendar.isleap(year):
        days_in_year = 366
    else:
        days_in_year = 365
    if day_number < 1 or day_number > days_in_year:
        return None
    date = datetime.date.fromordinal(datetime.date(year, 1, 1).toordinal() + day_number - 1)
    return date

# Apply function and create Date YYYY-MM-dd
accidents['Date'] = accidents.apply(lambda row: day_to_date(row['AccidentYear'], row['DayCount']), axis=1)


# Create matching date format with other dataset 
# YYYY-MM-ddThh:00+01:00  #Datum

# Combine the date and hour columns into a new column
accidents['Datum'] = pd.to_datetime(accidents['Date']) + pd.to_timedelta(accidents['AccidentHour'], unit='h')
accidents['Datum'] = accidents['Datum'].dt.strftime('%Y-%m-%dT%H:00+0100')



#%% Preprocessing and descriptive statistics
summary_accidents = accidents.describe(datetime_is_numeric = True)

## Accident type distribution

print(accidents['AccidentType_en'].value_counts())

# Get the value counts of 'AccidentType_en'
value_counts = accidents['AccidentType_en'].value_counts()

# Create a bar plot
plt.figure(figsize=(5,4)) 
value_counts.plot(kind='bar')
plt.xlabel('Accident Type')
plt.ylabel('Count')
plt.title('Accident Type Distribution')
for i, count in enumerate(value_counts):
    plt.annotate(str(count), xy=(i, count), ha='center', va='bottom')  #Show value counts
plt.show()



## Accident severity category distribution

print(accidents['AccidentSeverityCategory_en'].value_counts())

# Get the value counts of 'AccidentSeverityCategory_en'
value_counts = accidents['AccidentSeverityCategory_en'].value_counts()
# Create a bar plot
plt.figure(figsize=(5,4)) 
value_counts.plot(kind='bar')
plt.xlabel('Accident Severity Category')
plt.ylabel('Count')
plt.title('Accident Severity Category  Distribution')
for i, count in enumerate(value_counts):
    plt.annotate(str(count), xy=(i, count), ha='center', va='bottom')  #Show value counts
plt.show()



## Hours when accident happened distribution
print(accidents['AccidentHour'].value_counts())

# Get the value counts of 'AccidentSeverityCategory_en'
value_counts = accidents['AccidentHour'].value_counts()
# Create a bar plot
plt.figure(figsize=(10,6)) 
value_counts.plot(kind='bar')
plt.xlabel('Accident Hours')
plt.ylabel('Count')
plt.title('Accident Hours  Distribution')
for i, count in enumerate(value_counts):
    plt.annotate(str(count), xy=(i, count), ha='center', va='bottom')  #Show value counts
plt.show()

