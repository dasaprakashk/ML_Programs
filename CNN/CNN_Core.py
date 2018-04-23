#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:48:20 2018

@author: Das
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../MNIST_Data/train.csv', sep=',')

X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

X = np.array(X)
Y = np.array(Y)

for i in range(4):
    im = X[i, :].reshape([28, 28])
    plt.imshow(im)
    plt.show()
    
def padding(image, pad):
    return np.pad(image, [(0,0), (pad, pad), (pad, pad), (0,0)], mode='constant')

'''x = np.random.randn(4, 4, 4, 2)
x_pad = padding(x, 1)

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(x[0,:,:,0])
axarr[1].imshow(x_pad[0,:,:,0])'''

def conv_step(A_prev, W, b):
    z = np.sum(W * A_prev) + np.float(b)
    return z

def conv_forward(A_prev, W, b, hparameters):
    
    '''Previous layer's Tensor 
    m - No of input images
    n_H - Height of images
    n_W - Width of images
    n_C - No of channels'''
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    #Shape of the filter
    (f, f, n_C_prev, n_C) = W
    
    #Hyper parameters stride and padding
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    #Height and weight dimenstions of output 
    n_H = np.floor(1 + (n_H_prev + 2*pad - f)/stride)
    n_W = np.floor(1 + (n_W_prev + 2*pad - f)/stride)
    
    #Output tensor initialisation
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = padding(A_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = h*stride + f
                    h_start = w*stride
                    h_end = w*stride + f
                    
                    a_slice_prev = a_prev_pad[v_start:v_end, h_start:h_end]
                    
                    Z[i, h, w, c] = conv_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
    
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pooling_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = 1 + (n_H_prev - f) / stride
    n_W = 1 + (n_W_prev - f) / stride
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h*stride
                    v_end = h*stride + f
                    h_start = w*stride
                    h_end = w*stride + f
                    
                    a_prev_slice = A_prev[i, v_start:v_end, h_start:h_end, c]
                    
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.average(a_prev_slice)
                        
    cache = (A_prev, hparameters)
    return A, cache

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = 