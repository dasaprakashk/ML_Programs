#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:59:06 2018

@author: Das
"""
from ann_basic import ANNBasic
from ann_common import Common
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class ANN:
    def fit(self, X, Y, epoch):
        ann = ANNBasic()
        common = Common()
        w1, b1, w2, b2 = common.initialize_weights(X, Y, hidden_nodes=5)
        T = common.yEnc(Y)
        self.w1, self.b1, self.w2, self.b2, self.J = ann.backpropogation(X, Y, w1, b1, w2, b2, T, epoch)
        
    def predict(self, X):
        ann = ANNBasic()
        S, P = ann.feedforward(X, self.w1, self.b1, self.w2, self.b2)
        self.P = P
        return P
    
    def score(self, Y, P):
        common = Common()
        return common.accuracy(Y, P)
    
    def plot_cost(self):
        plt.plot(self.J, label='Training cost')
        plt.show()
        
common = Common()
X, Y = common.get_tabular_data('ecommerce_data.csv')
X = common.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=28, shuffle=True)

model = ANN()
model.fit(X_train, y_train, epoch=100000)
P = model.predict(X_train)
Yhat = np.argmax(P, axis=1)
accuracy = model.score(y_train, P) 

P_test = model.predict(X_test)
Y_test = np.argmax(P_test, axis=1)
accuracy_test = model.score(y_test, P_test)

model.plot_cost()
