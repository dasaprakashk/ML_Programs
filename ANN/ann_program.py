#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:59:06 2018

@author: Das
"""
from ann_basic import ANN_Basic
from ann_sgd import ANN_SGD
from ann_batchGD import ANN_BatchGD
from ann_common import Common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ANN:
    def fit(self, X, Y, epoch, activation, alpha, lambda_reg, optimizer):
        common = Common()
        w1, b1, w2, b2 = common.initialize_weights(X, Y, hidden_nodes=50)
        T = common.yEnc(Y)
        if optimizer is 'MBGD':
            ann = ANN_BatchGD()
        elif optimizer is 'SGD':
            ann = ANN_SGD()
        else:
            ann = ANN_Basic()
        self.w1, self.b1, self.w2, self.b2, self.J, self.A = ann.backpropagation(X, Y, w1, b1, w2, b2, T, epoch, activation, alpha, lambda_reg)
        
    def predict(self, X, activation, optimizer):
        if optimizer is 'MBGD':
            ann = ANN_BatchGD()
        elif optimizer is 'SGD':
            ann = ANN_SGD()
        else:
            ann = ANN_Basic()
        S, P = ann.feedforward(X, self.w1, self.b1, self.w2, self.b2, activation)
        self.P = P
        return P
    
    def score(self, Y, P):
        common = Common()
        return common.accuracy(Y, P)
    
    def plot_cost(self):
        plt.plot(self.J, label='Training cost')
        plt.show()
        
    def plot_accuracy(self):
        plt.plot(self.A, label='Accuracy')
        plt.show()
        
    #Get Data
    def get_tabular_data(self, filename):
        df = pd.read_csv(filename, sep=',')
        X = df.iloc[:, 0:-1]
        Y = df.iloc[:, -1]
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
 
common = Common()
model = ANN()       

#For ecommerce dataset
X, Y = model.get_tabular_data('ecommerce_data.csv')
X = common.normalize(X)

#For MNIST Dataset
'''
df = pd.read_csv('../../MNIST_Data/train.csv', sep = ',')
Y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X = X / 255.0'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=28, shuffle=True)

lambda_reg = 0
alpha = 1e-4
epoch = 5000
activation = 'sigmoid'
model.fit(X_train, y_train, epoch, activation, alpha, lambda_reg, optimizer=None)
P = model.predict(X_train, activation, optimizer=None)
Yhat = np.argmax(P, axis=1)
accuracy = model.score(y_train, P) 

P_test = model.predict(X_test, activation, optimizer=None)
Y_test = np.argmax(P_test, axis=1)
accuracy_test = model.score(y_test, P_test)

model.plot_cost()
model.plot_accuracy()
