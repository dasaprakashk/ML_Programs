#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:29:19 2018

@author: Das
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

class LogisticRegression:
    
    def mapfeature(self, X, order):
        poly = PolynomialFeatures(order)
        return poly.fit_transform(X)
    
    def fit(self, X, Y, W, b, alpha, lambda_rate, epochs):
        train = Train()
        self.W, self.b, self.J = train.gradient_descent(X, Y, W, b, alpha, lambda_rate, epochs)
        return self.W, self.b, self.J
    
    def predict(self, X):
        train = Train()
        return train.sigmoid(X, self.W, self.b)
    
    def plotData(self, X, Y):
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=118, alpha=0.5, cmap='coolwarm')
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')
        plt.show()
        
    def plotDecisionBoundary(self, X, Y, W, b, order):
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=118, alpha=0.5, cmap='coolwarm')
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')
        dim = np.linspace(-1, 1.5, 1000)
        x, y = np.meshgrid(dim, dim)
        poly = self.mapfeature(np.column_stack((x.flatten(), y.flatten())), order)
        z = (np.dot(poly, W) + b).reshape(1000, 1000)
        plt.contour(x, y, z, levels=[0], colors=['r'])
        plt.show()
        
class Train:
    def sigmoid(self, X, W, b):
        z = np.dot(X, W) + b
        return 1 / (1 + np.exp(-z))
    
    def gradient_descent(self, X, Y, W, b, alpha, lambda_rate, epochs):
        J = []
        for i in range(epochs):
            P = self.sigmoid(X, W, b)
            W = W * (1 - lambda_rate*alpha/Y.size)
            W = W - alpha * np.dot(X.T, P-Y)
            b = b - alpha * np.sum((P-Y), axis=0)
            if i%100 == 0:
                cost = self.cost_function(Y, P, lambda_rate, W)
                J.append(cost)
                rate = self.score(Y, P)
                print('Epoch: ' + str(i), 'Cost: ' + str(cost), 'Accuracy: ' + str(rate))
        return W, b, J
            
            
    def cost_function(self, Y, P, lambda_rate, W):
        l1 = np.log(P)
        l2 = np.log(1-P)
        reg = lambda_rate*np.sum(W**2)/(2*Y.size)
        return -np.sum(Y*l1 + (1-Y)*l2)/Y.size + reg
    
    def score(self, Y, P):
        return 1 - np.mean(np.abs(Y - np.round(P)))
    

df = pd.read_csv('Logistic_Regression_Reg.txt', sep = ',', header=None)
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X= np.array(X)
Y = np.array(Y)

model = LogisticRegression()
model.plotData(X, Y)

X_poly = model.mapfeature(X, order=6)

W = np.zeros(X_poly.shape[1])
b = 0
alpha = 0.001
epochs = 800
lambda_rate = 10
W, b, J = model.fit(X_poly, Y, W, b, alpha, lambda_rate, epochs)
P = model.predict(X_poly)
model.plotDecisionBoundary(X, Y, W, b, order=6)

