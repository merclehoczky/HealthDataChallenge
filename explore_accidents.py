#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:02:10 2023

@author: mercedeszlehoczky
"""

import pandas as pd


# Import Zuerich city accidents dataset
accidents = pd.read_csv('data/RoadTrafficAccidentLocations.csv', header = 0)

# View an instance
accidents.loc[3, :]

# Drop duplicate columns, only keep English ones
accidents = accidents.drop(columns = ['AccidentType_de', 'AccidentType_fr', 'AccidentType_it', 
                      'AccidentSeverityCategory_de', 'AccidentSeverityCategory_fr', 'AccidentSeverityCategory_it',
                      'RoadType_de', 'RoadType_fr', 'RoadType_it',
                      'AccidentMonth_de', 'AccidentMonth_fr', 'AccidentMonth_it',
                      'AccidentWeekDay_de', 'AccidentWeekDay_fr', 'AccidentWeekDay_it'])

# View datatypes 
accidents.dtypes

