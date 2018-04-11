#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:10:53 2018

@author: Das
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

df = pd.read_csv('ex2data2.txt', sep=',', header=None)
df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X = np.array(X)
Y = np.array(Y)

for i in range(X.shape[1]):
    X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
    
model = Sequential()
model.add(Dense(10, activation='tanh', input_dim=2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=400)