#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:59:06 2018

@author: Das
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Get Data
df = pd.read_csv('ecommerce_data.csv', sep=',')
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]