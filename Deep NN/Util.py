#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 23:06:20 2018

@author: Das
"""

import numpy as np

class Util:
    #Sigmoid function s = 1 / (1+ exp(-z))
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)), z
    
    def sigmoid_prime(self, z):
        return -np.exp(-z) / (1 + np.exp(-z))**2
    
    def softmax(self, z):
        expA = np.exp(z)
        return expA / expA.sum(axis=1, keepdims=True), z
    
    def softmax_prime(self, z):
        s = self.softmax(z)
        return s*(1-s)
    
    def tanh(self, z):
        return np.tanh(z), z
    
    def tanh_prime(self, z):
        return 1 - np.tanh(z)**2
    
    def relu(self, z):
        return np.where(z>0, z, 0), z
        
    def relu_prime(self, z):
        return np.where(z>0, 1, 0)
    
    def yEnc(self, Y):
        col = len(set(Y))
        T = np.zeros((col, Y.size))
        T[Y, np.arange(Y.size)] = 1
        return T.T