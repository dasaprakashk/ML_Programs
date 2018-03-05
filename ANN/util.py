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
    
    def normalize(self, X):
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
        return X
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def softmax(self, expA):
        return expA / expA.sum(axis=1, keepdims=True)
    
    def relu(self, z):
        return max(0, z)
    
    def sigmoid_prime(self, s):
        return s*(1-s)
    
    def softmax_prime(self, s):
        return s*(1-s)
    
    def relu_prime(self, s):
        if s > 0:
            return 1
        else:
            return 0
    def z(self, x, w, b):
        return np.dot(x, w) + b
    
    def activate(self, activation, z):
        if activation == 'relu':
            return self.relu(z)
        elif activation == 'softmax':
            return self.softmax(z)
        else:
            return self.sigmoid(z)