#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:55:55 2018

@author: Das
"""

import numpy as np
from util import Util

class ANNCommon:
    
    #Get weights
    def initialize_weights(self, X, Y, hidden_nodes):
        self.X = X
        self.Y = Y
        #Set the node size for i/p, o/p and hidden layers
        input_nodes = X.shape[1]
        output_nodes = len(set(Y))
        
        #Convert multi-class targets to one-hot encoding
        T = np.zeros((Y.size, output_nodes))
        T[np.arange(Y.size), Y] = 1
        self.T = T
        
        #Initialize weights and bias
        w1 = np.random.randn(input_nodes, hidden_nodes)/np.sqrt(input_nodes)
        b1 = np.zeros(hidden_nodes)
        w2 = np.random.randn(hidden_nodes, output_nodes)/np.sqrt(hidden_nodes)
        b2 = np.zeros(output_nodes)
        return w1, b1, w2, b2

    #Feed forward
    def feedforward(self, X, w1, b1, w2, b2):
        util = Util()
        z = util.z(X, w1, b1)
        s = util.sigmoid(z)
        z = util.z(s, w2, b2)
        expA = np.exp(z)
        sm = util.softmax(expA)
        return s, sm
    
    def w2_derivative(self, T, P, H):
        return np.dot(H.T, (P - T))
    
    def b2_derivative(self, T, P):
        return (P - T).sum(axis=0)
    
    def w1_derivative(self, T, P, w2, H, X):
        return np.dot(X.T, np.dot((P - T), w2.T)*H*(1-H))
    
    def b1_derivative(self, T, P, w2, H):
        return (np.dot((P - T), w2.T)*H*(1-H)).sum(axis=0)
    
    def backpropogation(self, X, w1, b1, w2, b2, epoch):
        cost_history = []
        for j in range(epoch):
            H, P = self.feedforward(X, w1,b1, w2, b2)
            learning_rate = 1e-4
            if j%1000 == 0:
                cost = self.cost(self.T, P)
                rate = self.accuracy(self.Y, P)
                cost_history.append(cost)
                print("Epoch: " + str(j), " Cost: " + str(cost), " Accuracy: " + str(rate))
            w2 = w2 - learning_rate * self.w2_derivative(self.T, P, H)
            b2 = b2 - learning_rate * self.b2_derivative(self.T, P)
            w1 = w1 - learning_rate * self.w1_derivative(self.T, P, w2, H, X)
            b1 = b1 - learning_rate * self.b1_derivative(self.T, P, w2, H)
        return w1, b1, w2, b2, cost_history

    
    def cost(self, T, P):
        return -np.sum(T*(np.log(P)))
    
    def accuracy(self, Y, P):
        Yhat = np.argmax(P, axis=1)
        return np.mean(Y == Yhat)