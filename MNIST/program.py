#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:28:41 2018

@author: Das
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
    return X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(expA):
    return expA / expA.sum(axis=1, keepdims=True)

def feedforward(X, w1, b1, w2, b2):
    z = np.dot(X, w1) + b1
    s = sigmoid(z)
    expA = z.dot(w2) + b2
    return s, softmax(expA)

def backprop():
    pass

def cost_function(T, P):
    return -np.sum(T*np.log(P))

def classification_rate(Y, Yhat):
    return np.mean(Y == Yhat)

df = pd.read_csv('../../MNIST_Data/train.csv', sep = ',')

Y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)

hidden_nodes = 20
input_nodes = X_train.shape[1]
output_nodes = len(set(y_train))

T = np.zeros((y_train.size, output_nodes))
T[np.arange(y_train.size), y_train] = 1

w1 = np.random.randn(input_nodes, hidden_nodes)
b1 = np.random.randn(hidden_nodes)
w2 = np.random.randn(hidden_nodes, output_nodes)
b2 = np.random.randn(output_nodes)

for j in range(1000):
    H, P = feedforward(X_train, w1, b1, w2, b2)
    if j%100 == 0:
        yhat = np.argmax(P, axis=1)
        cost = cost_function(T, P)
        rate = classification_rate(y_train, yhat)
        print('Epoch: ' + str(j), 'Cost: ' + str(cost), 'Accuracy: ' + str(rate))
    
    w2 = w2 - np.sum(np.dot(H.T, (T-P)))
    b2 = b2 - np.sum((T-P), axis=0)
    w1 = w1 - np.sum(np.dot(X_train.T, np.dot((T-P), w2.T)*H*(1-H)))
    b1 = b1 - np.sum(np.dot((T-P), w2.T)*H*(1-H), axis = 0)

T_test = np.zeros((y_test.size, output_nodes))
T[np.arange(y_test.size), y_test] = 1
H_test, P_test = feedforward(X_test, w1, b1, w2, b2)
y_testh = np.argmax(P_test, axis=0)
cost_test = cost_function(T_test, P_test)
rate = classification_rate(y_test, y_testh)