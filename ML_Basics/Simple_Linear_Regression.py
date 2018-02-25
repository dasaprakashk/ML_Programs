#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:03:38 2018

@author: Das
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Gradient descent
def gradient_descent(X, Y, theta, alpha, num_iter):
    new_theta = np.zeros(X.shape[1])
    J = []
    for j in range(num_iter):
        for i in range(theta.size):
            new_theta[i] = new_theta[i] - (alpha/Y.size) * np.sum((np.dot(X, theta) - Y) * X[:, i])
        for i in range(theta.size):
            theta[i] = new_theta[i]
        if j % 10 == 0:
            J.append(cost_function(X, Y, theta))
            print(theta)
    return theta, J
            
#Cost Function
def cost_function(X, Y, theta):
    return np.sum((np.dot(X, theta.T) - Y) ** 2) / (2*Y.size)

#R squared error
def r2(Y, Yhat):
    ssres = (Y - Yhat)
    sstot = Y - np.mean(Y)
    return 1 - (ssres.dot(ssres) / sstot.dot(sstot))

#Plot data
def plotData(X, Y, Yhat):
    plt.scatter(X, Y, marker='x', color = 'red')
    plt.plot(X, Yhat)
    plt.xlim(4, 24)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()
    
#Plot cost function
def plot_cost_function(J):
    plt.plot(J)
    plt.xlabel('Iterations (in 10s)')
    plt.ylabel('Cost')
    plt.show()
    
#Load data
df = pd.read_csv('Simple_LR.txt', sep=',', header=None)
df.insert(1, 'A', np.ones(df.shape[0]))
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
X = np.array(X)
Y = np.array(Y)

#Initialization
theta = np.ones(X.shape[1])
num_iter = 1500
alpha = 0.01

#Find optimal theta values, predictions, error
theta, J_history = gradient_descent(X, Y, theta, alpha, num_iter)
Yhat = np.dot(X, theta)
plotData(X[:,0], Y, Yhat)
plot_cost_function(J_history)
rsquared = r2(Y, Yhat)

#Predict
#Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([[3.5, 1]], theta)
print('For population = 35,000, we predict a profit of ' + str(predict1*10000))
predict2 = np.dot([[7, 1]], theta);
print('For population = 70,000, we predict a profit of ' + str(predict2*10000))