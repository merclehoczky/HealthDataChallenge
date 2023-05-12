#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:02:10 2023

@author: mercedeszlehoczky
"""

import pandas as pd
import numpy as np
import datetime


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

# View datatypes 
accidents.dtypes

# View dates 
print(accidents['AccidentWeekDay'].unique())

print(accidents['AccidentHour'].unique())

print(accidents['AccidentHour_text'].unique())

accidents[['AccidentYear', 'AccidentMonth_en','AccidentWeekDay', 'AccidentWeekDay_en','AccidentHour_text']].head(20)

accidents[['AccidentYear', 'AccidentMonth_en','AccidentWeekDay', 'AccidentWeekDay_en','AccidentHour_text']].tail(20)

# Dataset is until 2022 december

# Only keep data in 2019

accidents = accidents[accidents.AccidentYear == 2019]

#### Create date fixing loop
# Create new column for date
accidents["DayCount"] = np.nan

# Reset indices             
accidents = accidents.reset_index(drop=True)           

# Initialise counter
count = 1    

# Loop though data and add day counter
for row in accidents.index:
    accidents.at[row, 'DayCount'] = count   #Set first day of the month to 1
    if (accidents.at[row, 'AccidentWeekDay'] != accidents.at[row+1, 'AccidentWeekDay']): # If the days change
        count = count + 1 # Add to the counter (change the dates)
        accidents.at[row+1, 'DayCount'] = count  #Add counter as day counter
         