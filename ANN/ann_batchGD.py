#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:28:10 2018

@author: Das
"""

import numpy as np
from ann_common import Common

class ANN_BatchGD:
    def __init__(self):
        self.common = Common()
        
    def feedforward(self, X, w1, b1, w2, b2, activation):
        z = self.common.z(X, w1, b1)
        if activation == 'tanh':
            s = np.tanh(z)
        elif activation == 'relu':
            s = self.common.relu(z)
        else:
            s = self.common.sigmoid(z)
        z = self.common.z(s, w2, b2)
        expA = np.exp(z)
        sm = self.common.softmax(expA)
        return s, sm
    
    def backpropagation(self, X, Y, w1, b1, w2, b2, T, epoch, activation, alpha, lambda_reg):
        cost_history = []
        acc_history = []
        batch_size = 100
        batches = X.shape[0]/batch_size
        for i in range(epoch):
            H, P = self.feedforward(X, w1,b1, w2, b2, activation) 
            for j in range(batches):
                x = X[j*batch_size : j*batch_size + batch_size, :]
                y = Y[j*batch_size: j*batch_size + batch_size]
                t = T[j*batch_size: j*batch_size + batch_size]
                H, P = self.feedforward(x, w1, b1, w2, b2, activation)
                #Apply l2 regularization
                w2 = w2 - lambda_reg*w2
                #Apply descent
                w2 = w2 - alpha * self.common.w2_derivative(t, P, H)
                b2 = b2 - alpha * self.common.b2_derivative(t, P)
                #Apply regularization for w1
                w1 = w1 - lambda_reg*w1
                #Apply descent
                w1 = w1 - alpha * self.common.w1_derivative(t, P, w2, H, x, activation)
                b1 = b1 - alpha * self.common.b1_derivative(t, P, w2, H, activation)
           
                if i%100 == 0:
                    cost = self.cost(t, P, w1, w2, lambda_reg)
                    rate = self.common.accuracy(y, P)
                    cost_history.append(cost)
                    acc_history.append(rate)
                    print('Epoch: ' + str(i), 'Cost: ' + str(cost), 'Accuracy: ' + str(rate))
        return w1, b1, w2, b2, cost_history, acc_history
    
    def cost(self, T, P, w1, w2, reg_factor):
        J = -np.sum(T*(np.log(P)))
        #Added for L2 regularization
        J =J + reg_factor*np.sum(np.dot(w1, w2)**2)
        return J