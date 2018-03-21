#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:50:46 2018

@author: Das
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    
class DNN_Core:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        
    def initialise_parameters(self):
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            W = np.random.randn(self.layer_dims[l-1], self.layer_dims[l])*0.01
            b = np.random.randn(self.layer_dims[l], 1)*0.01
            parameters['W' + str(l)] = W
            parameters['b' + str(l)] = b
        return parameters

    def linear_forward(self, A, W, b):
        z = np.dot(A, W) + b.T
        return z, (A, W, b)
    
    def apply_dropout(self, A, dropout):
        D = np.random.randn(A.shape[0], A.shape[1])
        D = D < dropout
        A = A * D
        A = A / dropout
        return A, D

    def activation_forward(self, A, W, b, activation, dropout):
        util = Util()
        Z, linear_cache = self.linear_forward(A, W, b)
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
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[0]
        dW = np.dot(A_prev.T, dZ)/m
        db = dZ.sum(axis=0, keepdims=True)/m
        dA_prev = np.dot(dZ, W.T)
        return dA_prev, dW, db.T

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
        if dropout_cache is not None:
            dA_prev = dA_prev * dropout_cache
            dA_prev = dA_prev / dropout
        return dA_prev, dW, db
    
    def get_dropout_cache(self, caches, position):
        if position > -1:
            linear_cache, activation_cache, dropout_cache = caches[position]
        else:
            dropout_cache = None
        return dropout_cache

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

    def update_parameters(self, parameters, gradients, alpha, reg_rate, m):
        layers = len(parameters) // 2
        
        for l in range(layers):
            parameters["W" + str(l+1)] += (alpha * reg_rate/m) * parameters["W" + str(l+1)]
            parameters["W" + str(l+1)] -= alpha * gradients["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= alpha * gradients["db" + str(l+1)]
        return parameters
    
    def cost_function(self, Yhat, Y, parameters, reg_rate):
        L = len(parameters)//2
        sum_weights = 0
        for l in range(L):
            sum_weights += np.sum(parameters["W" + str(l + 1)])
        J = (reg_rate / (2*Y.size)) * sum_weights
        J += -np.sum(Y*(np.log(Yhat)))
        return J
    
    def batchGD(self, X, Y, epochs, alpha, parameters, reg_rate, dropout, print_cost):
        util = Util()
        J_history = []
        Acc_history = []
        T = util.yEnc(Y)
        for i in range(epochs):
            A, caches = self.feed_forward(X, parameters, dropout)
            gradients = self.back_propogation(T, A, caches, dropout)
            parameters = self.update_parameters(parameters, gradients, alpha, reg_rate, Y.size)
            if print_cost:
                if i % 100 == 0:
                    J = self.cost_function(A, T, parameters, reg_rate)
                    J_history.append(J)
                    rate = self.accuracy(A, Y)
                    Acc_history.append(rate)
                    print('Iteration: ' + str(i), 'Cost: ' + str(J) + \
                             'Accuracy: ' + str(rate))
                self.plot_cost(J_history)
        return A, gradients, parameters
    
    def SGD(self, X, Y, epochs, alpha, parameters, batch_size, reg_rate, dropout, print_cost):
        util = Util()
        J_history = []
        T = util.yEnc(Y)
        for i in range(epochs):
            s = np.arange(X.shape[0])
            np.random.shuffle(s)
            X = X[s]
            T = T[s]
            for k in range(0, X.shape[0], batch_size):
                x = X[k:k+batch_size, :]
                t = T[k:k+batch_size, :]
                A, caches = self.feed_forward(x, parameters, dropout)
                gradients = self.back_propogation(t, A, caches, dropout)
                parameters = self.update_parameters(parameters, gradients, alpha, reg_rate, Y.size)
            if print_cost:
                if i % 100 == 0:
                    A, caches = self.feed_forward(X, parameters, dropout)
                    J = self.cost_function(A, T, parameters, reg_rate)
                    J_history.append(J)
                    rate = self.accuracy(A, Y)
                    print('Iteration: ' + str(i), 'Cost: ' + str(J), 'Accuracy: ' + str(rate))
                self.plot_cost(J_history)
        A, caches = self.feed_forward(X, parameters, dropout)
        return A, gradients, parameters
    
    def plot_cost(self, J_history):
        plt.plot(J_history)
        
    def accuracy(self, Yhat, Y):
        Yhat = np.argmax(Yhat, axis = 1)
        return np.mean(Y == Yhat)

class DNN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
    
    def fit(self, X, Y, epochs, alpha, reg_rate, dropout_prob, batch_type):
        core = DNN_Core(self.layer_dims)
        parameters = core.initialise_parameters()
        if batch_type == 'full':
            A, gradients, parameters = core.batchGD(X, Y, epochs, alpha, \
                                    parameters, reg_rate, dropout_prob, True)
        else:
            if batch_type == 'mini_batch':
                batch_size = 512
            else:
                batch_size = 1
            A, gradients, parameters = core.SGD(X, Y, epochs, alpha, parameters,\
                                    batch_size, reg_rate, dropout_prob, True)
        return A, gradients, parameters
    
    def predict(self, X, parameters, dropout_prob):
        core = DNN_Core(self.layer_dims)
        Yhat, caches = core.feed_forward(X, parameters, dropout_prob)
        return Yhat
    
    def accuracy(self, Yhat, Y):
        core = DNN_Core(self.layer_dims)
        return core.accuracy(Yhat, Y)
    
    

df = pd.read_csv('../../MNIST_Data/train.csv', sep = ',')
df = df.iloc[:10000, :]
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

X = np.array(X)
Y = np.array(Y)

X = X/255.0


layer_dims = [X.shape[1], 30, 20, len(set(Y))]
deepNN = DNN(layer_dims)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=28) 

Yhat, gradients, parameters = deepNN.fit(X_train, Y_train, 1500, 0.2, 0, 1, 'mini_batch')

accuracy = deepNN.accuracy(Yhat, Y_train)

Yhat_test = deepNN.predict(X_test, parameters, 0.8)

accuracy_test = deepNN.accuracy(Yhat_test, Y_test)