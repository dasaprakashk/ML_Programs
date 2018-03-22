#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:50:46 2018

@author: Das
"""

import numpy as np
import pandas as pd
from DNN_SGDOptimizer import DNN_SGDOptimizer
from DNN_RMSPropOptimizer import DNN_RMSPropOptimizer
from DNN_AdamOptimizer import DNN_AdamOptimizer
from DNN_Core import DNN_Core
from sklearn.model_selection import train_test_split

class DNN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
    
    def fit_SGD(self, X, Y, epochs, alpha, reg_type, reg_rate, batch_type, momentum, nesterov):
        sgd = DNN_SGDOptimizer(self.layer_dims)
        
        if batch_type == 'full':
            batch_size = X.shape[0]
        if batch_type == 'mini_batch':
            batch_size = 512
        else:
            batch_size = 1
        A, gradients, parameters = sgd.SGD(X, Y, epochs, alpha, batch_size,\
                                    reg_type, reg_rate, momentum, nesterov, True)
        return A, gradients, parameters
    
    def fit_RMSProp(self, X, Y, epochs, alpha, reg_type, reg_rate, batch_type, momentum):
        rmsprop = DNN_RMSPropOptimizer(self.layer_dims)
        
        if batch_type == 'full':
            batch_size = X.shape[0]
        if batch_type == 'mini_batch':
            batch_size = 512
        else:
            batch_size = 1
        A, gradients, parameters = rmsprop.RMSProp(X, Y, epochs, alpha, batch_size,\
                                    reg_type, reg_rate, momentum, True)
        return A, gradients, parameters
    
    def fit_Adam(self, X, Y, epochs, alpha, reg_type, reg_rate, batch_type, beta1, beta2):
        adam = DNN_AdamOptimizer(self.layer_dims)
        
        if batch_type == 'full':
            batch_size = X.shape[0]
        if batch_type == 'mini_batch':
            batch_size = 512
        else:
            batch_size = 1
        A, gradients, parameters = adam.Adam(X, Y, epochs, alpha, batch_size,\
                                    reg_type, reg_rate, beta1, beta2, True)
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

'''Yhat, gradients, parameters = deepNN.fit_SGD(X_train, Y_train, 1500, 1e-2, 'l2', 0.5, \
                                         'mini_batch', 0.9, False)

Yhat, gradients, parameters = deepNN.fit_RMSProp(X_train, Y_train, 1500, 1e-4, 'l2', 0.5, \
                                         'mini_batch', 0.9)'''

Yhat, gradients, parameters = deepNN.fit_Adam(X_train, Y_train, 1500, 1e-4, 'l2', 0.5, \
                                         'mini_batch', 0.9, 0.999)

Yhat = deepNN.predict(X_train, parameters, 1)
                                       
accuracy = deepNN.accuracy(Yhat, Y_train)

Yhat_test = deepNN.predict(X_test, parameters, 1)

accuracy_test = deepNN.accuracy(Yhat_test, Y_test)