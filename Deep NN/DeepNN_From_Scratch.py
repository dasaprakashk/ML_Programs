#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:50:46 2018

@author: Das
"""

import numpy as np
import pandas as pd
from DNN_SGDOptimizer import DNN_Optimizer
from DNN_Core import DNN_Core
from sklearn.model_selection import train_test_split

class DNN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
    
    def fit(self, X, Y, epochs, alpha, reg_type, reg_rate, batch_type, momentum, nesterov):
        sgd = DNN_Optimizer(self.layer_dims)
        
        if batch_type == 'full':
            batch_size = X.shape[0]
        if batch_type == 'mini_batch':
            batch_size = 512
        else:
            batch_size = 1
        A, gradients, parameters = sgd.SGD(X, Y, epochs, alpha, batch_size,\
                                    reg_type, reg_rate, momentum, nesterov, True)
        return A, gradients, parameters
    
    def predict(self, X, parameters, dropout_prob):
        core = DNN_Core(self.layer_dims)
        Yhat, caches = core.feed_forward(X, parameters, dropout_prob)
        return Yhat
    
    def accuracy(self, Yhat, Y):
        core = DNN_Core(self.layer_dims)
        return core.accuracy(Yhat, Y)
    
    

df = pd.read_csv('../../MNIST_Data/train.csv', sep = ',')
df = df.iloc[:10000, :]
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

X = np.array(X)
Y = np.array(Y)

X = X/255.0


layer_dims = [X.shape[1], 30, 20, len(set(Y))]
deepNN = DNN(layer_dims)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=28) 

Yhat, gradients, parameters = deepNN.fit(X_train, Y_train, 1500, 0.1, 'l2', 0.9, \
                                         'mini_batch', 0.01, False)

accuracy = deepNN.accuracy(Yhat, Y_train)

Yhat_test = deepNN.predict(X_test, parameters, 1)

accuracy_test = deepNN.accuracy(Yhat_test, Y_test)