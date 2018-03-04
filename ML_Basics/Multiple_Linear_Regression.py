#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:16:26 2018

@author: Das
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        train = Train(X, Y)
        self.theta = np.zeros(X.shape[1])
        alpha = 0.01
        num_iter = 1500
        self.theta, self.cost_history = train.gradient_descent(self.theta, alpha, num_iter)
    
    def cost(self):
        train = Train(self.X, self.Y)
        return train.cost_function(self.theta)
    
    def plot_Cost(self):
        train = Train(self.X, self.Y)
        train.plot_cost_function(self.cost_history)
    
    def predict(self, X):
        test = Test(X)
        return test.get_predictions(self.theta)
    
    def score(self, Y, Y_pred):
        test = Test(X)
        return test.score(Y, Y_pred)
    
class Train:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    #Gradient descent
    def gradient_descent(self, theta, alpha, num_iter):
        new_theta = np.zeros(theta.size)
        J_history = []
        for j in range(num_iter):
            for i in range(theta.size):
                new_theta[i] = new_theta[i] - (alpha/self.Y.size)*np.sum((np.dot(self.X, theta.T) - self.Y) * self.X[:,i])
            for i in range(theta.size):
                theta[i] = new_theta[i]
            if j%100 == 0:
                J_history.append(self.cost_function(theta))
        return theta, J_history
        
    #Cost Function    
    def cost_function(self, theta):
        return (np.sum(np.dot(self.X, theta.T) - self.Y)**2)/(2*self.Y.size)
    
    #Plot cost function
    def plot_cost_function(self, J):
        plt.plot(J)
        plt.xlabel('Iterations (in 10s)')
        plt.ylabel('Cost')
        plt.show()

class Test:
    def __init__(self, X):
        self.X = X
        
    def get_predictions(self, theta):
        return np.dot(self.X, theta.T)
    
    #R-squated error
    def score(self, Y, Yhat):
        n = Y-Yhat
        d = Y-Y.mean()
        return 1 - (n.dot(n) / d.dot(d))
    
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

#Create model, predictions and error
model = LinearRegression()
model.fit(X, Y)
print("Training Cost: ", model.cost())
model.plot_Cost()
Yhat = model.predict(X)
print("Training Score: ", model.score(Y, Yhat))

#Prediction using Normal Equations to compare
thetaNE = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
YhatNE = np.dot(X, thetaNE.T)
print("Normal Equations Training Score: ", model.score(Y, YhatNE))