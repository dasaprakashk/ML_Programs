#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:37:08 2018

@author: Das
"""

import numpy as np
from ann_common import Common

class ANN_SGD:
    def __init__(self):
        self.common = Common()
        
    def feed_forward(self, X, w1, b1, w2, b2):
        z = self.common.z(X, w1, b1)
        s = np.tanh(z)
        expA = np.exp(self.common.z(s, w2, b2))
        sm = self.common.softmax(expA)
        return s, sm
    
    def backpropogation(self, X, Y, w1, b1, w2, b2, T, epoch, alpha, lambda_reg):
        #logging params
        cost_history = []
        #Set hyperparameters
        for i in range(epoch):
            H, P = self.feed_forward(X, w1, b1, w2, b2)
            if i%100 == 0:
                cost = self.cost(T, P, w1, w2, lambda_reg)
                rate = self.common.accuracy(Y, P)
                cost_history.append(cost)
                print('Epoch: ' + str(i), 'Cost: ' + str(cost), 'Accuracy: ' + str(rate))
            for j in range(X.shape[0]):
                #Apply l2 regularization
                w2 = w2 - lambda_reg*w2
                #Apply descent
                w2 = w2 - alpha * self.common.w2_derivative(T, P, H)
                b2 = b2 - alpha * self.common.b2_derivative(T, P)
                #Apply regularization for w1
                w1 = w1 - lambda_reg*w1
                #Apply descent
                w1 = w1 - alpha * self.common.w1_derivative(T, P, w2, H, X)
                b1 = b1 - alpha * self.common.b1_derivative(T, P, w2, H)
        return w1, b1, w2, b2, cost_history
    
    def cost(self, T, P, w1, w2, reg_rate):
        J = -np.sum(T * np.log(P))
        #Apply regularization
        J = J + reg_rate * np.sum(np.dot(w1, w2) ** 2)
        return J
        