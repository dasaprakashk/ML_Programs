#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:59:06 2018

@author: Das
"""

import pandas as pd
import numpy as np

def sigmoid(X, w, b):
    z = np.dot(X, w) + b
    return 1 / (1 + np.exp(-z))

def softmax(X, w1, b1, w2, b2):
    z = sigmoid(X, w1, b1)
    expA = np.exp(z.dot(w2) + b2)
    return expA / expA.sum(axis=1, keepdims=True)

#Get Data
df = pd.read_csv('ecommerce_data.csv', sep=',')
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]

X = np.array(X)
Y = np.array(Y)

hidden_nodes = 5
input_nodes = X.shape[1]
output_nodes = len(set(Y))
w1 = np.random.randn(input_nodes, hidden_nodes)
b1 = np.zeros(hidden_nodes)
w2 = np.random.randn(hidden_nodes, output_nodes)
b2 = np.zeros(output_nodes)

def feedforward(X, w1, b1, w2, b2):
    return softmax(X, w1, b1, w2, b2)

P = feedforward(X, w1,b1, w2, b2)
Yhat = np.argmax(P, axis=1)

score = np.mean(Y == Yhat)

