#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 23:43:51 2018

@author: Das
"""

import numpy as np
from Util import Util
from DNN_Core import DNN_Core

class DNN_AdamOptimizer:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
    
    #Stochastic Gradient descent - Generalized for Mini batch and batch
    def Adam(self, X, Y, epochs, alpha, batch_size, reg_type, reg_rate, beta1, beta2, print_cost):
        util = Util()
        core = DNN_Core(self.layer_dims)
        J_history = []
        T = util.yEnc(Y)
        parameters = core.initialise_parameters()
        v1 = core.initialise_velocity(parameters)
        v2 = core.initialise_velocity(parameters)
        if reg_type == 'l2':
            l2 = reg_rate
            dropout = 1
        elif reg_type == 'dropout':
            l2 = 0
            dropout = reg_rate
        for i in range(epochs):
            s = np.arange(X.shape[0])
            np.random.shuffle(s)
            X = X[s]
            T = T[s]
            for k in range(0, X.shape[0], batch_size):
                x = X[k:k+batch_size, :]
                t = T[k:k+batch_size, :]
                A, caches = core.feed_forward(x, parameters, dropout)
                gradients = core.back_propogation(t, A, caches, dropout)
                parameters = self.update_parameters(parameters, gradients, alpha, \
                                                    l2, v1, v2, beta1, beta2, Y.size)
            if print_cost:
                if i % 100 == 0:
                    AL, caches = core.feed_forward(X, parameters, dropout)
                    J = core.cost_function(AL, T, parameters, reg_rate)
                    J_history.append(J)
                    rate = core.accuracy(AL, Y)
                    print('Iteration: ' + str(i), 'Cost: ' + str(J), 'Accuracy: ' + str(rate))
                core.plot_cost(J_history)
        return A, gradients, parameters
    
    def update_parameters(self, parameters, gradients, alpha, l2, v, s, beta1, beta2, m):
        layers = len(parameters) // 2
        epsilon = 1e-8
        for l in range(layers):
            #l2 regularization
            parameters["W" + str(l+1)] += (alpha * l2/m) * parameters["W" + str(l+1)]
            
            v["W" + str(l+1)] = beta1 * v["W" + str(l+1)] + (1-beta1) * gradients["dW" + str(l+1)]
            v["b" + str(l+1)] = beta1 * v["b" + str(l+1)] + (1-beta1) * gradients["db" + str(l+1)]
            s["W" + str(l+1)] = beta2 * s["W" + str(l+1)] + (1-beta2) * gradients["dW" + str(l+1)]**2
            s["b" + str(l+1)] = beta2 * s["b" + str(l+1)] + (1-beta2) * gradients["db" + str(l+1)]**2
            parameters["W" + str(l+1)] -= (alpha * v["W" + str(l+1)]) / (np.sqrt(s["W" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] -= (alpha * v["b" + str(l+1)]) / (np.sqrt(s["b" + str(l+1)]) + epsilon)
        return parameters