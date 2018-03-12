#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:51:23 2018

@author: Das
"""

import numpy as np
import matplotlib.pyplot as plt

class Common:
    
    def sigmoid_plot():
        sample_z = np.linspace(-10, 10, 100)
        sample_a = 1 / (1 + np.exp(-sample_z))
        plt.xlim(-10, 10)
        plt.grid()
        plt.plot(sample_z, sample_a)
    
    def normalize(self, X):
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
        return X
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def softmax(self, expA):
        return expA / expA.sum(axis=1, keepdims=True)
    
    def relu(self, z):
        return np.where(z > 0, z, 0.0)
    
    def sigmoid_prime(self, z):
        return (np.exp(-z))/((1+np.exp(-z))**2)
    
    def softmax_prime(self, s):
        return s*(1-s)
    
    def relu_prime(self, s):
        return np.where(s > 0, 1.0, 0.0)
    
    def z(self, x, w, b):
        return np.dot(x, w) + b
    
    def activate(self, activation, z):
        if activation == 'relu':
            return self.relu(z)
        elif activation == 'softmax':
            return self.softmax(z)
        else:
            return self.sigmoid(z)
        
    #Initialize weights
    def initialize_weights(self, X, Y, hidden_nodes):
        self.X = X
        self.Y = Y
        #Set the node size for i/p, o/p and hidden layers
        input_nodes = X.shape[1]
        output_nodes = len(set(Y))
        
        #Initialize weights and bias
        w1 = np.random.randn(input_nodes, hidden_nodes)/np.sqrt(input_nodes)
        b1 = np.zeros(hidden_nodes)
        w2 = np.random.randn(hidden_nodes, output_nodes)/np.sqrt(hidden_nodes)
        b2 = np.zeros(output_nodes)
        return w1, b1, w2, b2
    
    def yEnc(self, Y):
        #Convert multi-class targets to one-hot encoding
        enc = len(set(Y))
        T = np.zeros((Y.size, enc))
        T[np.arange(Y.size), Y] = 1
        return T
    
    def w2_derivative(self, T, P, H):
        return np.dot(H.T, (P - T))
    
    def b2_derivative(self, T, P):
        return (P - T).sum(axis=0)
    
    def w1_derivative(self, T, P, w2, H, X, activation):
        H_prime = self.get_hprime(T, P, w2, H, activation)
        return np.dot(X.T, H_prime)
    
    def b1_derivative(self, T, P, w2, H, activation):
        return self.get_hprime(T, P, w2, H, activation).sum(axis=0)
    
    def get_hprime(self, T, P, w2, H, activation):
        if activation == 'tanh':
            return np.dot((P - T), w2.T) * (1 - H * H)
        elif activation == 'relu':
            return np.dot((P - T), w2.T) * self.relu_prime(H)
        else:
            return np.dot((P - T), w2.T) * self.softmax_prime(H)
    
    def accuracy(self, Y, P):
        Yhat = np.argmax(P, axis=1)
        return np.mean(Y == Yhat)
