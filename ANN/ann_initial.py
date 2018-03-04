#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:59:06 2018

@author: Das
"""

import pandas as pd
import numpy as np
    

def softmax(expA):
    return expA / expA.sum(axis=1, keepdims=True)

#Get Data
def get_data():
    df = pd.read_csv('ecommerce_data.csv', sep=',')
    X = df.iloc[:, 0:-1]
    Y = df.iloc[:, -1]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def feedforward(X, w1, b1, w2, b2):
    z = np.dot(X, w1) + b1
    s = 1 / (1 + np.exp(-z))
    expA = np.exp(s.dot(w2) + b2)
    return s, softmax(expA)

X, Y = get_data()
hidden_nodes = 5
input_nodes = X.shape[1]
output_nodes = len(set(Y))
T = np.zeros((Y.size, output_nodes))
T[np.arange(Y.size), Y] = 1
w1 = np.random.randn(input_nodes, hidden_nodes)
b1 = np.zeros(hidden_nodes)
w2 = np.random.randn(hidden_nodes, output_nodes)
b2 = np.zeros(output_nodes)

cost_history = []
for epoch in range(100000):
    H, P = feedforward(X, w1,b1, w2, b2)
    learning_rate = 0.0001
    if epoch%100 == 0:
        cost = np.sum(T*np.log(P))
        Yhat = np.argmax(P, axis=1)
        rate = np.mean(Y == Yhat)
        cost_history.append(cost)
        print("Classification rate: ", rate, "Cost: ", cost)
        
    w2 = w2 - learning_rate * -(np.dot(H.T, (T - P)))
    b2 = b2 - learning_rate * -(T - P).sum(axis=0)
    w1 = w1 - learning_rate * -np.dot(X.T, np.dot((T-P), w2.T)*H*(1 - H))
    b1 = b1 - learning_rate * -(np.dot((T-P), w2.T)*H*(1 - H)).sum(axis=0)
