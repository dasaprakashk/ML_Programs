#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:17:26 2018

@author: Das
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logistic_Regression:
    def fit(self, X, Y, theta, alpha, num_iter):
        train = Train()
        self.theta, J = train.gradient_descent(X, Y, theta, alpha, num_iter)
        return self.theta, J
    
    def predict(self, X):
        train = Train()
        return train.sigmoid(X, self.theta)
    
    def plot_Decision_Boundary(self, X, Y):
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5, cmap='coolwarm')
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.legend()
        x = np.linspace(-2, 2, 100)
        y = -(self.theta[0] * x + self.theta[2]) / self.theta[1]
        plt.plot(x, y, 'r')

class Train:
    #sigmoid = 1 / 1+e-z
    def sigmoid(self, X, theta):
        z = np.dot(X, theta.T)
        return 1 / (1 + np.exp(-z))
    
    def gradient_descent(self, X, Y, theta, alpha, num_iter):
        J = []
        for i in range(num_iter):
            P = self.sigmoid(X, theta)
            theta = theta - alpha * np.dot(X.T, (Y-P))/Y.size
            if i%100 == 0:
                cost = self.cost_function(Y, P)
                rate = self.score(Y, P)
                print('Iteration: ' + str(i), 'Cost: ' + str(cost), 'Accuracy: ' + str(rate))
                J.append(cost)
        return theta, J
    
    #cross entropy J = -Sigma(Ylog(P) + (1-Y)log(1-P))
    def cost_function(self, Y, P):
        l1 = np.log(P)
        l2 = 1- np.log(P)
        return np.sum(Y*l1 + (1-Y)*l2)/Y.size
    
    def score(self, Y, P):
        return np.mean(np.abs(Y - np.round(P)))
    
def normalize(X):
    for i in range(X.shape[1]-1):
        X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
    return X
    
df = pd.read_csv('Logistic_Regression.txt', sep=',', header=None)
df.insert(2, 'A', np.ones(df.shape[0]))
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X = np.array(X)
Y = np.array(Y)

X = normalize(X)

model = Logistic_Regression()
theta = np.zeros(3)
alpha = 0.01
num_iter = 400
theta, J = model.fit(X, Y, theta, alpha, num_iter)
P = model.predict(X)
model.plot_Decision_Boundary(X, Y)