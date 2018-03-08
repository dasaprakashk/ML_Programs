#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:55:55 2018

@author: Das
"""

import numpy as np
from ann_common import Common

class ANNBasic:
    
    def __init__(self):
        self.common = Common()
    
    #Feed forward
    def feedforward(self, X, w1, b1, w2, b2):
        z = self.common.z(X, w1, b1)
        s = self.common.sigmoid(z)
        z = self.common.z(s, w2, b2)
        expA = np.exp(z)
        sm = self.common.softmax(expA)
        return s, sm
    
    def backpropogation(self, X, Y, w1, b1, w2, b2, T, epoch):
        cost_history = []
        reg_factor = 1e-4
        learning_rate = 1e-4
        for j in range(epoch):
            H, P = self.feedforward(X, w1,b1, w2, b2)            
            if j%1000 == 0:
                cost = self.cost(T, P, w1, w2, reg_factor)
                rate = self.common.accuracy(Y, P)
                cost_history.append(cost)
                print("Epoch: " + str(j), " Cost: " + str(cost), " Accuracy: " + str(rate))
            #Added for derivative of l2 regularization
            w2 = w2 - (reg_factor*w2)
            #derivative of cost function w.r.t w2
            w2 = w2 - learning_rate * self.common.w2_derivative(T, P, H)
            #derivative of cost function w.r.t b2
            b2 = b2 - learning_rate * self.common.b2_derivative(T, P)
            #Added for derivative of l2 regularization
            w1 = w1 - (reg_factor*w1)
            #derivative of cost function w.r.t w1
            w1 = w1 - learning_rate * self.common.w1_derivative(T, P, w2, H, X)
            #derivative of cost function w.r.t b1
            b1 = b1 - learning_rate * self.common.b1_derivative(T, P, w2, H)
        return w1, b1, w2, b2, cost_history
    
    def cost(self, T, P, w1, w2, reg_factor):
        J = -np.sum(T*(np.log(P)))
        #Added for L2 regularization
        J =J + reg_factor*np.sum(np.dot(w1, w2)**2)
        return J