#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:18:44 2018

@author: Das
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs

#Cross Entropy
def computeCost(theta, X, Y):
    Yhat = sigmoid(X, theta)
    J = (-1)*(np.dot(Y, np.log(Yhat)) + np.dot((1-Y), np.log(1-Yhat)))/Y.size
    return J.flatten()

#calculate weights
def computeGradient(theta, X, Y):
    new_theta = np.zeros(X.shape[1])
    Yhat = sigmoid(X, theta)
    for i in range(theta.size):
        new_theta[i] = np.sum((Yhat-Y)*X[:,i])/Y.size
    for i in range(theta.size):
        theta[i] = new_theta[i]
    return theta
    
#Sigmoid a = 1 / (1 + exp(X.T*theta))
def sigmoid(X, theta):
    z = np.dot(X, theta.T)
    return 1 / (1 + np.exp(-z))

#Simplify cost function to pass to BFGS
def costBFGS(theta):
    return computeCost(theta, X, Y) 

#Simplify weight function to pass to BFHS
def gradientBFGS(theta):
    return computeGradient(theta, X, Y)

#Load and prepare data
df = pd.read_csv('Logistic_regression.txt', sep=',', header=None)
df.insert(df.shape[1]-1, 'Bias', np.ones(df.shape[0]))
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
X = np.array(X)
Y = np.array(Y)

#Weights and cost of initial theta
theta = np.zeros(X.shape[1])
J = computeCost(theta, X, Y)
grad = computeGradient(theta, X, Y)

#Weights and cost
theta = np.array([0.2, 0.2, -24])
J = computeCost(theta, X, Y)
grad = computeGradient(theta, X, Y)

#Weights by BFGS and predictions 
theta = fmin_bfgs(costBFGS, x0=theta, maxiter=400, fprime=gradientBFGS)
J = computeCost(theta, X, Y)
Yhat = sigmoid(X, theta)

#Final classification rate
print("Final classification rate:", 1 - np.abs(Y - np.round(Yhat)).sum() / Y.size)

