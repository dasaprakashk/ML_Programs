#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 23:07:26 2018

@author: Das

Has core functions for computational model of Deep Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
from Util import Util

class DNN_Core:
    #Init
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
    
    #Initialise weights and bias of all layers
    def initialise_parameters(self):
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            W = np.random.randn(self.layer_dims[l-1], self.layer_dims[l])*0.01
            b = np.random.randn(self.layer_dims[l], 1)*0.01
            parameters['W' + str(l)] = W
            parameters['b' + str(l)] = b
        return parameters
    
    #Initialise velocity for momentum and optimizers
    def initialise_velocity(self, parameters):
        L = len(parameters)//2
        velocity = {}
        for l in range(L):
            velocity["W" + str(l+1)] = 0
            velocity["b" + str(l+1)] = 0
        return velocity
    
    #Apply dropout for feedforward
    def apply_dropout(self, A, dropout):
        #Create a dropout matrix with same dimension as activation layer matrix
        D = np.random.randn(A.shape[0], A.shape[1])
        
        #Scale  down using the dropout ratio parameter
        D = D < dropout
        
        #Apply to activation matrix
        A = A * D
        
        #Scale up activation layer
        A = A / dropout
        return A, D

    #Linear approximation function for feed forward
    def linear_forward(self, A, W, b):
        z = np.dot(A, W) + b.T
        return z, (A, W, b)

    #Activation function for feed forward
    def activation_forward(self, A, W, b, activation, dropout):
        util = Util()
        
        #Get linear approximation
        Z, linear_cache = self.linear_forward(A, W, b)
        
        #Call activation function appropriately
        if activation == 'relu':
            A, activation_cache = util.relu(Z)
        elif activation == 'tanh':
            A, activation_cache = util.tanh(Z)
        elif activation == 'softmax':
            A, activation_cache = util.softmax(Z)
        else:
            A, activation_cache = util.sigmoid(Z)
        if dropout == None:
            dropout_cache = None
        else:
            A, dropout_cache = self.apply_dropout(A, dropout)
        return A, (linear_cache, activation_cache, dropout_cache)

    #Feed Forward
    def feed_forward(self, X, parameters, dropout):
        caches = []
        L = len(parameters) // 2
        A = X
        for l in range(1, L):
            A_prev = A
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            A, cache = self.activation_forward(A_prev, W, b, 'relu', dropout)
            caches.append(cache)
        W = parameters["W" + str(L)]
        b = parameters["b" + str(L)]
        A, cache = self.activation_forward(A, W, b, 'softmax', None)
        caches.append(cache)
        return A, caches
    
    #Backpropogation - Linear approximation function
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[0]
        dW = np.dot(A_prev.T, dZ)/m
        db = dZ.sum(axis=0, keepdims=True)/m
        dA_prev = np.dot(dZ, W.T)
        return dA_prev, dW, db.T
    
    #Backpropogation - Activation function
    def activation_backward(self, dA, cache, dropout_cache, activation, dropout):
        util = Util()
        linear_cache, activation_cache, d = cache
        A_prev, W, b = linear_cache
        if activation == 'relu':
            dZ = dA * util.relu_prime(activation_cache)
        elif activation == 'tanh':
            dZ = dA * util.tanh_prime(activation_cache)
        elif activation == 'softmax':
            dZ = dA
        else:
            dZ = dA * util.sigmoid_prime(activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        #Apply dropout to hidden layer
        if dropout_cache is not None:
            dA_prev = dA_prev * dropout_cache
            dA_prev = dA_prev / dropout
        return dA_prev, dW, db
    
    #Backpropogation
    def back_propogation(self, Y, Yhat, caches, dropout):
        L = len(caches)
        gradients = {}
        dA = Yhat-Y
        current_cache = caches[L-1]
        dropout_cache = self.get_dropout_cache(caches, L-2)
        dA_prev, dW, db = self.activation_backward(dA, current_cache, dropout_cache, 'softmax', dropout)
        gradients['dA' + str(L-1)] = dA_prev
        gradients['dW' + str(L)] = dW
        gradients['db' + str(L)] = db
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dropout_cache = self.get_dropout_cache(caches, l-1)
            dA_prev, dW, db = self.activation_backward(dA_prev, current_cache, dropout_cache, 'relu', dropout)
            gradients['dA' + str(l)] = dA_prev
            gradients['dW' + str(l + 1)] = dW
            gradients['db' + str(l + 1)] = db
        return gradients
    
    #Cost function - Negative log likelihood
    def cost_function(self, Yhat, Y, parameters, reg_rate):
        L = len(parameters)//2
        sum_weights = 0
        for l in range(L):
            sum_weights += np.sum(parameters["W" + str(l + 1)])
        J = (reg_rate / (2*Y.size)) * sum_weights
        J += -np.sum(Y*(np.log(Yhat)))
        return J
    
    def get_dropout_cache(self, caches, position):
        if position > -1:
            linear_cache, activation_cache, dropout_cache = caches[position]
        else:
            dropout_cache = None
        return dropout_cache
    
    def plot_cost(self, J_history):
        plt.plot(J_history)
        
    def accuracy(self, Yhat, Y):
        Yhat = np.argmax(Yhat, axis = 1)
        return np.mean(Y == Yhat)