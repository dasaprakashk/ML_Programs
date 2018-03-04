#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:59:06 2018

@author: Das
"""
from ann_common import ANNCommon
from util import Util
import matplotlib.pyplot as plt
import numpy as np

class ANN:
    def fit(self, X, Y, epoch=100000):
        ann = ANNCommon()
        w1, b1, w2, b2 = ann.initialize_weights(X, Y, hidden_nodes=5)
        self.w1, self.b1, self.w2, self.b2, self.J = ann.backpropogation(X, w1, b1, w2, b2, epoch) 
        self.Y = Y
        
    def predict(self, X):
        ann = ANNCommon()
        S, P = ann.feedforward(X, self.w1, self.b1, self.w2, self.b2)
        self.P = P
        return P
    
    def score(self, Y, P):
        ann = ANNCommon()
        return ann.accuracy(self.Y, self.P)
    
    def plot_cost(self):
        plt.plot(self.J)
        plt.show()
        

X, Y = Util().get_tabular_data('ecommerce_data.csv')
model = ANN()
model.fit(X, Y)
P = model.predict(X)
Yhat = np.argmax(P, axis=1)
accuracy = model.score(Y, P) 



