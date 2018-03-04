#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:51:23 2018

@author: Das
"""

import numpy as np 
import pandas as pd

class Util:
    
    #Get Data
    def get_tabular_data(self, filename):
        df = pd.read_csv(filename, sep=',')
        X = df.iloc[:, 0:-1]
        Y = df.iloc[:, -1]
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    
    def softmax(self, expA):
        return expA / expA.sum(axis=1, keepdims=True)
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))