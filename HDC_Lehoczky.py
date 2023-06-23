#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:55:23 2023

@author: mercedeszlehoczky
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:14:25 2023

@author: mercedeszlehoczky
"""
import pandas as pd
import numpy as np
import datetime
import calendar
import matplotlib.pyplot as plt


#%% Zürich city accidents


# Enlarge output display capacity
pd.set_option('display.max_rows', 500)

# Set the display option to show all columns
pd.set_option('display.max_columns', None)

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

# View date structure 
print(accidents['AccidentWeekDay'].unique())

print(accidents['AccidentHour'].unique())

print(accidents['AccidentHour_text'].unique())

accidents[['AccidentYear', 'AccidentMonth_en','AccidentWeekDay', 'AccidentWeekDay_en','AccidentHour_text']].head(20)

accidents[['AccidentYear', 'AccidentMonth_en','AccidentWeekDay', 'AccidentWeekDay_en','AccidentHour_text']].tail(20)

# Dataset is until 2022 december

#%% Preprocessing

# Only keep data in 2019

accidents = accidents[accidents.AccidentYear == 2019]

#%%% Fix date formats

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
# 'Date'

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
# YYYY-MM-ddThh:00+01:00  
#'Datum'

# Combine the date and hour columns into a new column
accidents['Datum'] = pd.to_datetime(accidents['Date']) + pd.to_timedelta(accidents['AccidentHour'], unit='h')
accidents['Datum'] = accidents['Datum'].dt.strftime('%Y-%m-%dT%H:00+0100')



#%% Descriptive statistics
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



#%% Zürich city weather


# %% Import dataset
w19_hr = pd.read_csv('../data/2019/ugz_ogd_meteo_h1_2019.csv', header = 0)
w19_hr

# %% View datatypes
w19_hr.dtypes

# View measurement locations
print(w19_hr['Standort'].unique())

# View measurement parameters
print(w19_hr['Parameter'].unique())

# %% Filter parameters
conditions = ['StrGlo', 'WD', 'WVv']
filtered = w19_hr.loc[~w19_hr['Parameter'].isin(conditions)]
filtered.head()

# %% Average hourly measurements
# Group by date (hours)
 # Create new dataframe with locations sorted under timepoints (long to wide)

w19_new = pd.pivot(filtered, index = ['Datum', 'Standort'], 
                   columns = ['Parameter'], values = ['Wert'])

# View levels of indices on columns
w19_new.columns.levels
w19_new.columns.get_level_values(1)
# Drop outermost level (Wert)
w19_new.columns = w19_new.columns.droplevel() 

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

#View
w19.head(50)



#%% Merge datasets 

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
df_19['Is_Rainy'] = df_19['RainDur'].astype(bool)


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




# %% Correlations

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


# %% Create variable which counts the number of accidents per day   
# 'EntryCount'
# Group the dataset by day and count the number of occurrences per day
daily_count = df_19.groupby(df_19['Date'].dt.date).size().reset_index(name='EntryCount')

# Merge the count of occurrences with the original dataset
merged_df = pd.merge(df_19, daily_count, left_on=df_19['Date'].dt.date, right_on=daily_count['Date'], how='left')
merged_df = merged_df.drop(columns=['key_0', 'Date_y']).rename(columns={'Date_x': 'Date'})

# Convert EntryCount to integer
merged_df['EntryCount'] = merged_df['EntryCount'].astype(int)



# %% Select columns of interest
columns_of_interest = ['AccidentType_en', 'AccidentSeverityCategory_en',
                       'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                       'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                       'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 'Is_Rainy',
                       'Hr', 'RainDur', 'T', 'WVs', 'p']

# %% Create selection df with columns of interest and EntryCount
df_select = merged_df[columns_of_interest + ['EntryCount']]



#%% Compute correlation matrix

# Compute the correlation matrix
corr_matrix = df_select[['EntryCount'] + columns_of_interest].corr()


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
# Initiating a fig and axis object
fig, ax = plt.subplots(figsize=(18, 16))
# Create a plot
cax = ax.imshow(pearson_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
# Set axis tick labels
ax.set_xticks(ticks=range(len(pearson_matrix.columns)))
ax.set_xticklabels(pearson_matrix.columns, rotation=90)
ax.set_yticks(ticks=range(len(pearson_matrix.columns)))
ax.set_yticklabels(pearson_matrix.columns)
# Resize the tick parameters
ax.tick_params(axis="both", labelsize=8)
# Adding a color bar
fig.colorbar(cax).ax.tick_params(labelsize=8)
# Add annotation
for (x, y), t in np.ndenumerate(pearson_matrix):
    ax.text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)

plt.title('Pearson correlations')
plt.show()

# With numeric only 
# Select only numeric variables for Pearson correlation
numeric_variables = df_select.select_dtypes(include=[np.number])

# Compute Pearson correlation matrix
pearson_matrix = numeric_variables.corr(method='pearson')

# Plot Pearson correlation matrix
fig, ax = plt.subplots(figsize=(18, 16))
cax = ax.imshow(pearson_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
ax.set_xticks(ticks=range(len(pearson_matrix.columns)))
ax.set_xticklabels(pearson_matrix.columns, rotation=90)
ax.set_yticks(ticks=range(len(pearson_matrix.columns)))
ax.set_yticklabels(pearson_matrix.columns)
ax.tick_params(axis="both", labelsize=8)
fig.colorbar(cax).ax.tick_params(labelsize=8)
for (x, y), t in np.ndenumerate(pearson_matrix):
    ax.text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)

plt.title('Pearson correlations with numeric variables')
plt.show()

#%% Spearman's correlation for nonlinear associations
spearman_matrix = df_select.corr(method='spearman')

# Spearman's plot
# Initiating a fig and axis object
fig, ax = plt.subplots(figsize=(18, 16))
# Create a plot
cax = ax.imshow(spearman_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
# Set axis tick labels
ax.set_xticks(ticks=range(len(spearman_matrix.columns)))
ax.set_xticklabels(spearman_matrix.columns, rotation=90)
ax.set_yticks(ticks=range(len(spearman_matrix.columns)))
ax.set_yticklabels(spearman_matrix.columns)
# Resize the tick parameters
ax.tick_params(axis="both", labelsize=8)
# Adding a color bar
fig.colorbar(cax).ax.tick_params(labelsize=8)
# Add annotation
for (x, y), t in np.ndenumerate(spearman_matrix):
    ax.text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)

plt.title('Spearman correlations')
plt.show()

# With numeric only
# Select only numeric variables for Spearman correlation
numeric_variables = df_select.select_dtypes(include=[np.number])

# Compute Spearman correlation matrix
spearman_matrix = numeric_variables.corr(method='spearman')

# Plot Spearman correlation matrix
fig, ax = plt.subplots(figsize=(18, 16))
cax = ax.imshow(spearman_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
ax.set_xticks(ticks=range(len(spearman_matrix.columns)))
ax.set_xticklabels(spearman_matrix.columns, rotation=90)
ax.set_yticks(ticks=range(len(spearman_matrix.columns)))
ax.set_yticklabels(spearman_matrix.columns)
ax.tick_params(axis="both", labelsize=8)
fig.colorbar(cax).ax.tick_params(labelsize=8)
for (x, y), t in np.ndenumerate(spearman_matrix):
    ax.text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)

plt.title('Spearman correlations with numeric only')
plt.show()


# %% Pearson's and Spearmen's numeric side by side

# Select only numeric variables for Pearson correlation
numeric_variables = df_select.select_dtypes(include=[np.number])

# Compute Pearson correlation matrix
pearson_matrix = numeric_variables.corr(method='pearson')

# Plot Pearson correlation matrix
fig, axes = plt.subplots(figsize=(18, 16), ncols=2)
cax1 = axes[0].imshow(pearson_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
axes[0].set_xticks(ticks=range(len(pearson_matrix.columns)))
axes[0].set_xticklabels(pearson_matrix.columns, rotation=90)
axes[0].set_yticks(ticks=range(len(pearson_matrix.columns)))
axes[0].set_yticklabels(pearson_matrix.columns)
axes[0].tick_params(axis="both", labelsize=8)
fig.colorbar(cax1, ax=axes[0]).ax.tick_params(labelsize=8)
for (x, y), t in np.ndenumerate(pearson_matrix):
    axes[0].text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)
axes[0].set_title('Pearson correlations')

# Select only numeric variables for Spearman correlation
numeric_variables = df_select.select_dtypes(include=[np.number])

# Compute Spearman correlation matrix
spearman_matrix = numeric_variables.corr(method='spearman')

# Plot Spearman correlation matrix
cax2 = axes[1].imshow(spearman_matrix.values, interpolation='nearest', cmap='Blues', vmin=-1, vmax=1)
axes[1].set_xticks(ticks=range(len(spearman_matrix.columns)))
axes[1].set_xticklabels(spearman_matrix.columns, rotation=90)
axes[1].set_yticks(ticks=range(len(spearman_matrix.columns)))
axes[1].set_yticklabels(spearman_matrix.columns)
axes[1].tick_params(axis="both", labelsize=8)
fig.colorbar(cax2, ax=axes[1]).ax.tick_params(labelsize=8)
for (x, y), t in np.ndenumerate(spearman_matrix):
    axes[1].text(y, x, "{:.2f}".format(t), ha='center', va='center', fontsize=8)
axes[1].set_title('Spearman correlations')

# Adjust the layout
plt.tight_layout()
# Show the plot
plt.show()

#%% Cramer’s V for categoricals

# Using association_metrics library
import association_metrics as am
# Select columns of Category type
selected_columns = ['AccidentType_en', 'AccidentSeverityCategory_en',
                  'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                  'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                  'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 'Is_Rainy',
                  'EntryCount']

# Convert selected columns to category data type
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

boolean_vars = ['Is_Holiday', 'Is_Weekend', 'Is_RushHour', 'Is_Rainy',
                'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle']
pbc_matrix = []

for var in boolean_vars:
    r, p = pointbiserialr(df_select[var], df_select['EntryCount'])
    pbc_matrix.append((var, r, p))

pbc_matrix = pd.DataFrame(pbc_matrix, columns=['Variable', 'Correlation', 'p-value'])



# %% INFERENTIAL

import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats



#%% LINEAR REGRESSION
# with https://realpython.com/linear-regression-in-python/
# from https://datatofish.com/multiple-linear-regression-python/

# with OLS
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Define the columns of interest
columns_of_interest = ['AccidentType_en', 'AccidentSeverityCategory_en',
                       'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                       'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                       'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 'Is_Rainy',
                       'Hr', 'RainDur', 'T', 'WVs', 'p', 'EntryCount']

# Extract the columns of interest from the DataFrame
df_regression = df_select[columns_of_interest]


# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_regression, drop_first=True)

# Handle missing values by imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df_encoded)

## Create the feature matrix X and target variable y
X = X[:, :-1]  # Exclude the target variable from the feature matrix
y = df_encoded['EntryCount'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regr.predict(X_test)


# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)

# Retrieve the coefficients
coefficients_ols = regr.coef_

# Print the coefficients
for i, coef in enumerate(coefficients_ols):
    print(f'Coefficient {i}: {coef}')

# %% LASSO
from sklearn.linear_model import Lasso

# Create the Lasso regression model
lasso = Lasso(alpha=0.5)  # Set the regularization parameter alpha

# Fit the Lasso regression model
lasso.fit(X_train, y_train)

# Predict on the test set
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# %% LASSO with grid search


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the alpha values to be tested
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

# Create the Lasso regression model
lasso = Lasso()

# Perform grid search
param_grid = {'alpha': alphas}
grid_search = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best alpha and best model
best_alpha = grid_search.best_params_['alpha']
best_model = grid_search.best_estimator_

# Fit the best model
best_model.fit(X_train, y_train)

# Print the best alpha and best model
print("Best Alpha:", best_alpha)
print("Best Model:", best_model)

# Evaluate the model
mse = np.mean((best_model.predict(X_test) - y_test) ** 2)
print("Mean Squared Error:", mse)

# %% Manual models

#%% Short model 1: weather conditions only
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Create the feature matrix X and target variable y
X_short1 = df_encoded[['Is_Rainy', 'RainDur', 'T', 'WVs']]
y = df_select['EntryCount']


# Create an instance of LinearRegression
model = LinearRegression()

# Fit the model
model.fit(X_short1, y)

# Get the predicted values
y_pred = model.predict(X_short1)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Retrieve the coefficients and column names
coefficients = model.coef_
column_names = X_short1.columns

# Create a DataFrame to store the coefficients
coefficients_df = pd.DataFrame({'Variable': column_names, 'Coefficient': coefficients})

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Coefficients:")
print(coefficients_df)


#%% Short model 2: holiday, weekend and rush hour

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Create the feature matrix X and target variable y
X_short2 = df_encoded[['Is_Holiday', 'Is_Weekend', 'Is_RushHour']]
y = df_select['EntryCount']


# Create an instance of LinearRegression
model = LinearRegression()

# Fit the model
model.fit(X_short2, y)

# Get the predicted values
y_pred = model.predict(X_short2)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Retrieve the coefficients and column names
coefficients = model.coef_
column_names = X_short2.columns

# Create a DataFrame to store the coefficients
coefficients_df = pd.DataFrame({'Variable': column_names, 'Coefficient': coefficients})

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Coefficients:")
print(coefficients_df)

#%% Long model
# Long model: weather conditions, weekend, holidays and rush hour

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Create the feature matrix X and target variable y
X_long = df_encoded[['Is_Rainy', 'RainDur', 'T', 'WVs',
               'Is_Holiday', 'Is_Weekend', 'Is_RushHour']]
y = df_select['EntryCount']


# Create an instance of LinearRegression
model = LinearRegression()

# Fit the model
model.fit(X_long, y)

# Get the predicted values
y_pred = model.predict(X_long)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Retrieve the coefficients and column names
coefficients = model.coef_
column_names = X_long.columns

# Create a DataFrame to store the coefficients
coefficients_df = pd.DataFrame({'Variable': column_names, 'Coefficient': coefficients})

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Coefficients:")
print(coefficients_df)


