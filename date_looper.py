#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:30:06 2023

@author: mercedeszlehoczky
"""


import pandas as pd
import numpy as np
import datetime
import calendar

### Create date fixing loop
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

    

