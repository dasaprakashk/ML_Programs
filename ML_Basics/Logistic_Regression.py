#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:18:44 2018

@author: Das
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs

class LogisticRegression:
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.theta = np.array([0.2, 0.2, -24])
        train = Train(X, Y)
        self.theta = train.gradient_descent(self.theta)
        
    def predict(self, X):
        train = Train(self.X, self.Y)
        return train.sigmoid(X, self.theta)
    
    def score(self, Y, Yhat):
        return 1 - np.abs(Y - np.round(Yhat)).sum() / Y.size
    
class Train:
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    #Sigmoid a = 1 / (1 + exp(X.T*theta))
    def sigmoid(self, X, theta):
        z = np.dot(self.X, theta.T)
        return 1 / (1 + np.exp(-z))
        
    #Cross Entropy
    def computeCost(self, theta):
        Yhat = self.sigmoid(self.X, theta)
        J = (-1)*(np.dot(self.Y, np.log(Yhat)) + np.dot((1-self.Y), np.log(1-Yhat)))/self.Y.size
        return J.flatten()

    #calculate weights
    def computeGradient(self, theta):
        new_theta = np.zeros(self.X.shape[1])
        Yhat = self.sigmoid(self.X, theta)
        for i in range(theta.size):
            new_theta[i] = np.sum((Yhat-self.Y)*self.X[:,i])/self.Y.size
        for i in range(theta.size):
            theta[i] = new_theta[i]
        return theta

    def gradient_descent(self, theta):
        return fmin_bfgs(self.computeCost, x0=theta, maxiter=400, fprime=self.computeGradient)

#Load and prepare data
df = pd.read_csv('Logistic_regression.txt', sep=',', header=None)
df.insert(df.shape[1]-1, 'Bias', np.ones(df.shape[0]))
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
X = np.array(X)
Y = np.array(Y)

model = LogisticRegression()
model.fit(X, Y)
Yhat = model.predict(X)
print("Classification Rate: ", model.score(Y, Yhat))

