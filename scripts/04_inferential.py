#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:48:43 2023

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

#%% LINEAR REGRESSION
# with https://realpython.com/linear-regression-in-python/
#from https://datatofish.com/multiple-linear-regression-python/

# with OLS
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

# Select columns of interest
columns_of_interest = ['AccidentType_en', 'AccidentSeverityCategory_en',
                       'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                       'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                       'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 
                       'Hr', 'RainDur', 'T', 'WVs', 'p']

# Extract the columns of interest from the DataFrame
df_regression = df_select[columns_of_interest]

# Convert relevant columns to category dtype
df_regression['AccidentType_en'] = df_regression['AccidentType_en'].astype('category')
df_regression['AccidentSeverityCategory_en'] = df_regression['AccidentSeverityCategory_en'].astype('category')
df_regression['RoadType_en'] = df_regression['RoadType_en'].astype('category')
df_regression['AccidentWeekDay_en'] = df_regression['AccidentWeekDay_en'].astype('category')
df_regression['AccidentHour'] = df_regression['AccidentHour'].astype('category')

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_regression, drop_first=True)

# Handle missing values by imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df_encoded)

# Split the data into predictors (X) and target variable (y)
y = df_select['EntryCount'].values

# Fit the linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Obtain the coefficients and p-values
coefficients = regr.coef_
p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues[1:]

# Create a DataFrame to store the coefficients and p-values
result_mreg = pd.DataFrame({'Variable': df_encoded.columns, 'Coefficient': coefficients, 'p-value': p_values})

# Add a column for significance
result_mreg['Significant'] = result_mreg['p-value'] < 0.05






#%% LASSO
# with https://machinelearningmastery.com/lasso-regression-with-python/

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

# Select columns of interest
columns_of_interest = ['AccidentType_en', 'AccidentSeverityCategory_en',
                       'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                       'RoadType_en', 'AccidentWeekDay_en', 'AccidentHour',
                       'Is_Holiday', 'Is_Weekend', 'Is_RushHour', 
                       'Hr', 'RainDur', 'T', 'WVs', 'p']

# Extract the columns of interest from the DataFrame
df_regression = df_select[columns_of_interest]

# Convert relevant columns to category dtype
df_regression['AccidentType_en'] = df_regression['AccidentType_en'].astype('category')
df_regression['AccidentSeverityCategory_en'] = df_regression['AccidentSeverityCategory_en'].astype('category')
df_regression['RoadType_en'] = df_regression['RoadType_en'].astype('category')
df_regression['AccidentWeekDay_en'] = df_regression['AccidentWeekDay_en'].astype('category')
df_regression['AccidentHour'] = df_regression['AccidentHour'].astype('category')

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_regression, drop_first=True)

# Handle missing values by imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df_encoded)

# Split the data into predictors (X) and target variable (y)
y = df_select['EntryCount'].values

# Create and fit the LASSO model
lasso = Lasso(alpha=0.1)  # Adjust the alpha value as desired
lasso.fit(X, y)

# Obtain the coefficients and p-values
coefficients = lasso.coef_
#FIX THIS:
p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues[1:]

# Create a DataFrame to store the coefficients and p-values
result_lasso = pd.DataFrame({'Variable': df_encoded.columns, 'Coefficient': coefficients, 'p-value': p_values})

# Add a column for significance
result_lasso['Significant'] = result_lasso['p-value'] < 0.05



#%% F-test

import statsmodels.api as sm
from sklearn.linear_model import Lasso
from scipy import stats

# Fit the OLS model
model_ols = sm.OLS(y, X)
results_ols = model_ols.fit()

# Fit the LASSO model
model_lasso = Lasso(alpha=0.1)  # Adjust the alpha parameter as needed
model_lasso.fit(X, y)

# Calculate the sum of squared residuals for both models
rss_ols = ((results_ols.resid) ** 2).sum()
rss_lasso = ((y - model_lasso.predict(X)) ** 2).sum()

# Perform the F-test
f_value = ((rss_ols - rss_lasso) / (X.shape[1])) / (rss_lasso / (X.shape[0] - X.shape[1]))
p_value = 1 - stats.f.cdf(f_value, X.shape[1], X.shape[0] - X.shape[1])

# Print the F-test results
print("F-value:", f_value)
print("p-value:", p_value)


