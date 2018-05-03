#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 22:01:58 2018

@author: Das
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py


#Constants
CONST_NUM_CLASSES = 6
CONST_NUM_EPOCHS = 10
CONST_BATCH_SIZE = 128
CONST_SEED = 101
NUM_CHANNELS = 3
IMAGE_SIZE = 64

def load_dataset():
    ds_train = h5py.File('../../Handsigns/train_signs.h5', 'r')
    X_train = np.array(ds_train['train_set_x'][:])
    y_train = np.array(ds_train['train_set_y'][:])

    ds_test = h5py.File('../../Handsigns/test_signs.h5', "r")
    X_test = np.array(ds_test["test_set_x"][:]) # your test set features
    y_test = np.array(ds_test["test_set_y"][:]) # your test set labels
    
    classes = np.array(ds_test["list_classes"][:]) # the list of classes
    
    return X_train, y_train, X_test, y_test, classes

def visualise_data(X):
    for i in np.random.choice(X.shape[0], 5):
        plt.imshow(X[i])
        plt.show()
        
X_train, y_train, X_test, y_test, classes = load_dataset()
visualise_data(X_train)

#Normalize
X_train = X_train/255.0
X_test = X_test/255.0

#Train & validation sets
X_train, y_train, X_valid, y_valid = train_test_split(X_train, test_size = 0.05, shuffle=True)

#One-hot vectorizing
T_train = np.zeros([y_train.size, 6])
T_train[np.arange(y_train.size), y_train] = 1

T_valid = np.zeros([y_valid.size, 6])
T_valid[np.arange(y_valid.size), y_valid] = 1

T_test = np.zeros([y_test.size, 6])
T_test[np.arange(y_test.size), y_test] = 1

#TF placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='X')
y = tf.placeholder(dtype=tf.int32, shape=[None, CONST_NUM_CLASSES], name='labels')
is_train = tf.placeholder(dtype=tf.bool, name='training_phase')

#get weights
def get_filters(channel_in, channel_out, kernel):
    shape = [kernel[0], kernel[1], channel_in, channel_out]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, seed=CONST_SEED, name='W'))

#get bias
def get_bias(channel_out):
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[channel_out], name='b'))

#convolution
def conv_forward(X, channel_out, kernel, strides, padding, name):
    shape = X.get_shape().as_list()
    channel_in = shape[-1]
    filtr = get_filters(channel_in, channel_out, kernel)
    return tf.nn.conv2d(X, filtr, strides, padding, name=name)

#batch_normalization
def batch_normalization(X, channel_out, is_conv=True):
    beta = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[channel_out]), 
                       trainable=True, name='beta')
    gamma = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[channel_out]), 
                        trainable=True, name='gamma')
    
    if is_conv:
        batch_mean, batch_var = tf.nn.moments(X, [0,1,2])
    else:
        batch_mean, batch_var = tf.nn.moments(X, [0])
        
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
    def update_mean_var():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_train, update_mean_var, lambda:(ema.average(batch_mean), ema.average(batch_var)))
    norm = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3, name='batch_norm')
    return norm

def activation_forward(X):
    shape = X.get_shape().as_list()
    channel_out = shape[-1]
    return tf.nn.relu(tf.nn.bias_add(X, channel_out))
