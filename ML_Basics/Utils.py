#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:51:26 2018

@author: Das
"""

import numpy as np

class utils:
    #sigmoid function sigma = - 1 / 1 + exp(-z)
    def sigmoid(X, theta):
        z = np.dot(X, theta.T)
        return 1 / (1 + np.exp(-z))
    
    
    def softmax(a):
        expA = np.exp(a)
        return expA / expA.sum(axis=1, keep_dims=True)
    
    
    