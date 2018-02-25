#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:16:26 2018

@author: Das
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Gradient descent
def gradient_descent(X, Y, theta, alpha, num_iter):
    new_theta = np.zeros(theta.size)
    J_history = []
    for j in range(num_iter):
        for i in range(theta.size):
            new_theta[i] = new_theta[i] - (alpha/Y.size)*np.sum((np.dot(X, theta.T) - Y) * X[:,i])
        for i in range(theta.size):
            theta[i] = new_theta[i]
        if j%100 == 0:
            J_history.append(cost_function(X, Y, theta))
    return theta, J_history
        
#Cost Function    
def cost_function(X, Y, theta):
    return (np.sum(np.dot(X, theta.T) - Y)**2)/(2*Y.size)

#R-squated error
def r2(Y, Yhat):
    n = Y-Yhat
    d = Y-Y.mean()
    return 1 - (n.dot(n) / d.dot(d))

#Plot cost function
def plot_cost_function(J):
    plt.plot(J)
    plt.xlabel('Iterations (in 10s)')
    plt.ylabel('Cost')
    plt.show()
    
#Get Data
df = pd.read_csv('Multiple_LR.txt', sep=',', header=None)
df.insert(2, 'A', np.ones(df.shape[0]))
X = df.iloc[:, 0:-1]
Y = df.iloc[:,-1]
X = np.array(X)
Y = np.array(Y)

#Generalization
for i in range(X.shape[1] - 1):
    X[:, i] = (X[:, i] - X[:, i].mean())/X[:, i].std()

#Initialization of parameters
theta = np.zeros(X.shape[1])
alpha = 0.01
num_iter = 1500

#Creaye model, predictions and error
theta, J_history = gradient_descent(X, Y, theta, alpha, num_iter)
Yhat = np.dot(X, theta.T)
rsquared = r2(Y, Yhat)

#Plot cost function
plot_cost_function(J_history)

#Prediction using Normal Equations to compare
thetaNE = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
YhatNE = np.dot(X, thetaNE.T)
rsquaredNE = r2(Y, YhatNE)