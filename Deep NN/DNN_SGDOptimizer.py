#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 23:43:51 2018

@author: Das
"""

import numpy as np
from Util import Util
from DNN_Core import DNN_Core

class DNN_Optimizer:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
    
    def SGD(self, X, Y, epochs, alpha, batch_size, reg_type, reg_rate, momentum, nesterov, print_cost):
        util = Util()
        core = DNN_Core(self.layer_dims)
        J_history = []
        T = util.yEnc(Y)
        parameters = core.initialise_parameters()
        velocity = core.initialise_velocity(parameters)
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
                                                    l2, velocity, momentum, nesterov, Y.size)
            if print_cost:
                if i % 100 == 0:
                    A, caches = core.feed_forward(X, parameters, dropout)
                    J = core.cost_function(A, T, parameters, reg_rate)
                    J_history.append(J)
                    rate = core.accuracy(A, Y)
                    print('Iteration: ' + str(i), 'Cost: ' + str(J), 'Accuracy: ' + str(rate))
                core.plot_cost(J_history)
        A, caches = core.feed_forward(X, parameters, dropout)
        return A, gradients, parameters
    
    def update_parameters(self, parameters, gradients, alpha, l2, velocity, momentum, nesterov, m):
        layers = len(parameters) // 2
        for l in range(layers):
            parameters["W" + str(l+1)] += (alpha * l2/m) * parameters["W" + str(l+1)]
            velocity["W" + str(l+1)] = momentum * velocity["W" + str(l+1)] - \
                                        alpha * gradients["dW" + str(l+1)]
            velocity["b" + str(l+1)] = momentum * velocity["b" + str(l+1)] - \
                                        alpha * gradients["db" + str(l+1)]
            if nesterov:
                parameters["W" + str(l+1)] += momentum * velocity["W" + str(l+1)] - \
                                                alpha * gradients["W" + str(l+1)]
                parameters["b" + str(l+1)] += momentum * velocity["b" + str(l+1)] - \
                                                alpha * gradients["b" + str(l+1)]
            else:
                parameters["W" + str(l+1)] += velocity["W" + str(l+1)]
                parameters["b" + str(l+1)] += velocity["b" + str(l+1)]
        return parameters