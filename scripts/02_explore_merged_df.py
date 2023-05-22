#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:36:40 2023

@author: mercedeszlehoczky
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd

df_19 = dd.merge(accidents, w19, how='inner', on=['Datum'])
