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

#%% Clean up data
# Drop NaNs, nonsense values
df_19.dtypes

# Fix datatypes
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

#%% Infter additional information from data

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


#%% Rush hour
# from https://www.tomtom.com/traffic-index/zurich-traffic/

df_19['Is_RushHour'] = np.nan
rushhour = [7,8,16,17,18]

df_19['Is_RushHour'] = df_19['AccidentHour'].isin(rushhour)

#%% Rain
# Indicate rain
df_19['Is_Rainy'] = np.where(df_19['RainDur'] != 0, "True", "False")

#%% Wind
print(df_19['WVs'].min())
print(df_19['WVs'].max())

#The wind is low strength, expected not to make a difference.



#%% Summary statistics
summary_df19 = df_19.describe()



#%% Plots, exploration

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

## Plot Number of accidents per day
# Set figure size
plt.figure(figsize=(12, 6))
# Histogram, for each day
plt.hist(df_19['Date'], bins=365)
# Set labels and title
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Distribution of number of accidents per day')
# Set the x-axis ticker to display month labels
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format month as abbreviated name
# Rotate xlabel for readability
plt.xticks(rotation=45)
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()


## Plot Number of accidents per day, weekends highlighted
# Set figure size
plt.figure(figsize=(12, 6))
# Histogram, for each day
plt.hist(df_19['Date'], bins=365)
# Set labels and title
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Distribution of number of accidents per day')
# Set the x-axis ticker to display month labels
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format month as abbreviated name
# Rotate xlabel for readability
plt.xticks(rotation=45)
# Highlight weekends
weekend_dates = df_19[df_19['Is_Weekend']]['Date']
for weekend_date in weekend_dates:
    plt.axvspan(weekend_date, weekend_date , color='lightgray', alpha=0.1)
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()


## Plot Number of accidents per day, Fridays highlighted
# Set figure size
plt.figure(figsize=(12, 6))
# Histogram, for each day
plt.hist(df_19['Date'], bins=365)
# Set labels and title
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Distribution of number of accidents per day, highlight on Fridays')
# Set the x-axis ticker to display month labels
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format month as abbreviated name
# Rotate xlabel for readability
plt.xticks(rotation=45)
# Highlight Fridays
friday_dates = df_19[df_19['AccidentWeekDay_en'] == 'Friday']['Date']
for friday_date in friday_dates:
    plt.axvspan(friday_date, friday_date, color='lightgray', alpha=0.1)
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()



## Plot Number of accidents per days of the week 
# Set figure size
plt.figure(figsize=(12, 6))
# Histogram, for each day
counts, bins, patches = plt.hist(df_19['AccidentWeekDay_en'], bins=7, rwidth=0.9)
# Set labels and title
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.title('Distribution of Number of Accidents per Day')
# Set the x-axis ticker to display day labels
plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# Add count on top of bars
for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width() / 2, count, str(int(count)),
             ha='center', va='bottom')
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()




## Plot accident type per month 

# Set figure size
plt.figure(figsize=(12, 6))
# Group the data by AccidentType_en and AccidentMonth, and count the occurrences
grouped_data = df_19.groupby(['AccidentType_en', 'AccidentMonth']).size().unstack()
# Plot the grouped bar chart
ax = grouped_data.plot(kind='bar', stacked=True, width=0.8)
# Set labels and title
plt.xlabel('Accident Type')
plt.ylabel('Count')
plt.title('Accident Types per Month')
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
# Move the legend to the right side
plt.legend(title='Month', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()





## Plot accident severity per month
# Set figure size
plt.figure(figsize=(12, 6))
# Group the data by severity and AccidentMonth, and count the occurrences
grouped_data = df_19.groupby(['AccidentSeverityCategory_en', 'AccidentMonth']).size().unstack()
# Plot the grouped bar chart
ax = grouped_data.plot(kind='bar', stacked=True, width=0.8)
# Set labels and title
plt.xlabel('Accident Type')
plt.ylabel('Count')
plt.title('Accident Severity Categories per Month')
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
# Move the legend to the right side
plt.legend(title='Month', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()




## Plot Accident types by road types
plt.figure(figsize=(12, 6))
# Group the data by AccidentType_en and RoadType, and count the occurrences
grouped_data = df_19.groupby(['AccidentType_en', 'RoadType_en']).size().unstack()
# Plot the grouped bar chart
ax = grouped_data.plot(kind='bar', stacked=True, width=0.8)
plt.xlabel('Accident Type')
plt.ylabel('Count')
plt.title('Accident Types by Road Types')
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
# Move the legend to the right side
plt.legend(title='Road Type', bbox_to_anchor=(1.02, 1), loc='upper left')
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()


## Plot Accidents by rain duration intervals
# Convert RainDur into 10-minute intervals
df_19['RainDurIntervals'] = pd.cut(df_19['RainDur'], bins=np.arange(0, 70, 10), right=False)
# Group the dataset by road type and rain duration intervals, and calculate the count for each occasion
grouped_data = df_19.groupby(['RoadType_en', 'RainDurIntervals']).size().reset_index(name='Count')
# Set the width of each bar
bar_width = 0.35
# Set the x-axis values based on the unique rain duration intervals
x = np.arange(len(grouped_data['RainDurIntervals'].unique()))
# Plot the count of occasions for each road type and rain duration interval
for i, road_type in enumerate(grouped_data['RoadType_en'].unique()):
    road_type_data = grouped_data[grouped_data['RoadType_en'] == road_type]
    plt.bar(x + i * bar_width, road_type_data['Count'], width=bar_width, label=road_type)

        # Add count labels on top of the bars
    for j, count in enumerate(road_type_data['Count']):
        plt.text(x[j] + i * bar_width, count, str(count), ha='center', va='bottom')
# Set the x-axis tick labels as the rain duration intervals
plt.xticks(x + (bar_width * (len(grouped_data['RoadType_en'].unique()) - 1)) 
                   / 2, grouped_data['RainDurIntervals'].unique())
plt.xlabel('Rain Duration Intervals (min)')
plt.ylabel('Count')
plt.title('Count of Occasions by Rain Duration Intervals and Road Type')
plt.legend()
# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=45)
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()



## Plot Accident severity by rain duration intervals
# Convert RainDur into 10-minute intervals
df_19['RainDurIntervals'] = pd.cut(df_19['RainDur'], bins=np.arange(0, 70, 10), right=False)
# Group the dataset by accident severity and rain duration intervals, and calculate the count for each occasion
grouped_data = df_19.groupby(['AccidentSeverityCategory_en', 'RainDurIntervals']).size().reset_index(name='Count')
# Set the width of each bar
bar_width = 0.35
# Set the x-axis values based on the unique rain duration intervals
x = np.arange(len(grouped_data['RainDurIntervals'].unique()))
# Plot the count of occasions for each accident severity and rain duration interval
for i, severity in enumerate(grouped_data['AccidentSeverityCategory_en'].unique()):
    severity_data = grouped_data[grouped_data['AccidentSeverityCategory_en'] == severity]
    plt.bar(x + i * bar_width, severity_data['Count'], width=bar_width, label=severity)
    
# Set the x-axis tick labels as the rain duration intervals
plt.xticks(x + (bar_width * (len(grouped_data['AccidentSeverityCategory_en'].unique()) - 1)) / 2, grouped_data['RainDurIntervals'].unique())
plt.xlabel('Rain Duration Intervals (min)')
plt.ylabel('Count')
plt.title('Count of Occasions by Rain Duration Intervals and Severity of Accident')
plt.legend()
# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=45)
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()


## Plot Number of accidents in light of temperature
# Set figure size
plt.figure(figsize=(12, 6))
# Histogram, for each day
counts, bins, patches = plt.hist(df_19['T'], bins=50, rwidth=0.9)
# Set labels and title
plt.xlabel('T (°C)')
plt.ylabel('Count')
plt.title('Distribution of Number of Accidents by Hourly Temperature')
# Add count on top of bars
for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width() / 2, count, str(int(count)),
             ha='center', va='bottom')
# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
plt.show()


## Plot Accident Severity by Temperature
# Define the temperature bins
temperature_bins = np.arange(-3, 36, 2)
# Group the data by temperature and accident severity, and count the occurrences
grouped_data = df_19.groupby([pd.cut(df_19['T'], temperature_bins), 'AccidentSeverityCategory_en']).size().unstack()
# Set figure size
plt.figure(figsize=(12, 6))
# Plot the grouped bar chart
ax = grouped_data.plot(kind='bar', stacked=True, width=0.8)
# Set labels and title
plt.xlabel('Temperature (°C)')
plt.ylabel('Count')
plt.title('Accident Severity by Temperature')
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
# Move the legend to the right side
plt.legend(title='Accident Severity', bbox_to_anchor=(1.02, 1), loc='upper left')
# Adjust the layout to prevent overlapping of labels
# plt.tight_layout()
plt.show()




