#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:35:01 2018

@author: Das
"""

import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(101)
tf.set_random_seed(101)

df = pd.read_csv('ex2data2.txt', sep=',', header=None)
df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X = np.array(X)
Y = np.array(Y)

Y = np.reshape(Y, (Y.size, 1))

for i in range(X.shape[1]):
    X[:, i] = (X[:, i] - np.mean(X[: ,i]))/np.std(X[:, i])

x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

input_nodes = 2
hidden_nodes = 10
output_nodes = 1

#hyperparameters
learning_rate = 0.03
epochs = 1500

W1 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes]))
b1 = tf.Variable(tf.zeros([1, hidden_nodes]))
W2 = tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
b2 = tf.Variable(tf.zeros([1, output_nodes]))

def linear_forward(A, W, b):
    mul = tf.matmul(A, W)
    z = tf.add(mul, b)
    return z

def activation_forward(A, W, b, activation):
    z = linear_forward(A, W, b)
    if activation == 'sigmoid':
        A = tf.nn.sigmoid(z)
    elif activation == 'relu':
        A = tf.nn.relu(z)
    elif activation == 'tanh':
        A = tf.nn.tanh(z)
    elif activation == 'softmax':
        A = tf.nn.softmax(z)
    return A

def feed_forward(x, W1, b1, W2, b2):
    A1 = activation_forward(x, W1, b1, 'tanh')
    A2 = activation_forward(A1, W2, b2, 'sigmoid')
    return A2

A = feed_forward(x, W1, b1, W2, b2)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=A))
opt = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)
accuracy = 1 - tf.reduce_mean(tf.abs(y - tf.round(A)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        _, c, acc, predictions = sess.run([opt, cost, accuracy, A], feed_dict={x:X, y:Y})
        if epoch%100 == 0:
            print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(c), "acc =", "{:.5f}".format(acc))