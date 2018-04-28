#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:06:15 2018

@author: Das
"""

import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Constants
CONST_NUM_EPOCHS = 5
CONST_BATCH_SIZE = 128
CONST_NUM_LABELS = 10
CONST_SEED = 101
NUM_CHANNELS = 3
IMAGE_SIZE = 32
IMAGES_PER_FILE = 10000
NUM_FILES = 5
FILE_DIR = "../../CIFAR-10"

def unpickle(file_dir, file):
    file = os.path.join(file_dir, file)
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_dataset():
    
    #Initialize train and test tensors
    X_train = np.zeros([NUM_FILES*IMAGES_PER_FILE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
    y_train = np.zeros([NUM_FILES*IMAGES_PER_FILE])
    X_test = np.zeros([IMAGES_PER_FILE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
    y_test = np.zeros([IMAGES_PER_FILE])
    
    #Load data from files
    for i in range(5):
        start = i*IMAGES_PER_FILE 
        end = i*IMAGES_PER_FILE + IMAGES_PER_FILE
        d = unpickle(FILE_DIR, "data_batch_" + str(i+1))
        data = np.array(d[b'data'])
        X_train[start:end] = np.reshape(data, [data.shape[0], 3, 32, 32])
        y_train[start:end] = d[b'labels']
    
    d = unpickle(FILE_DIR, "test_batch")
    d[b'data']
    X_test[0:IMAGES_PER_FILE] = np.reshape(data, [data.shape[0], 3, 32, 32])
    y_test[0:IMAGES_PER_FILE] = d[b'labels']
    
    #Transpose to move channels to end
    X_train = np.transpose(X_train, [0,2,3,1]).astype('float32')
    X_test = np.transpose(X_test, [0,2,3,1]).astype('float32')

    #Normalize
    X_train = X_train/255.0

    #Create validation_set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.05, shuffle=True)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#Visualisation
def display_pics(X):
    X = X.astype("uint8")
    fig, axes = plt.subplots(5,5, figsize=(3,3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes[j][k].set_axis_off()
            axes[j][k].imshow(X[i:i+1][0])


X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

train_size = X_train.shape[0]
test_size = X_test.shape[0]

T_train = np.zeros([y_train.size, CONST_NUM_LABELS])
T_train[np.arange(y_train), y_train] = 1

T_valid = np.zeros([y_valid.size, CONST_NUM_LABELS])
T_valid[np.arange(y_valid), y_valid] = 1

T_test = np.zeros([y_test.size, CONST_NUM_LABELS])
T_test[np.arange(y_test), y_test] = 1

display_pics(X_test)


def get_filter(channel_in, channel_out, filter_shape):
    shape = [filter_shape[0], filter_shape[1], channel_in, channel_out]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

def get_bias(channel_out, const=True):
    if const:
        return tf.Variable(tf.constant(0.1, shape=[channel_out]), name='b')
    else:
        return tf.Variable(tf.zeros([channel_out]), name='b')
    
def batch_norm(X, channel_out, is_train=True):
    beta = tf.Variable(tf.constant(0.0, shape=[channel_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[channel_out]), name='gamma', trainable=True)
    
    batch_mean, batch_var = tf.nn.moments(X, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    update_ma = ema.apply([batch_mean, batch_var])
    mean, var = tf.cond(is_train, update_ma,
                       lambda:(ema.average(batch_mean), ema.average(batch_var)))
    norm = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
    return norm
    
def conv_forward(X, channel_out, filter_shape, strides, padding, name, batch_norm=True, is_train=True):
    filtr = get_filter(X.shape[-1], channel_out, filter_shape)
    bias = get_bias(channel_out)
    conv = tf.nn.conv2d(X, filtr, strides, padding, name=name)
    if batch_norm:
        conv = batch_norm(conv, channel_out, is_train)
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu_'+name)
    return relu

def flatten(X):
    shape = tf.shape(X)
    return tf.reshape(X, [shape[0], shape[1]*shape[2]*shape[3]], name='flatten')

def fc_forward(X, channel_out, name, activation, is_train=True):
    W = tf.Variable(tf.truncated_normal([X.shape[-1], channel_out], stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, [channel_out]), name='b')
    if activation == 'relu':
        X = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W), b), name='relu_' + name)
        X = batch_norm(X, channel_out, is_train)
        if is_train:
            X = tf.nn.dropout(X, keep_prob=0.5, name='dropout_' + name)
    elif activation=='softmax':
        X = tf.nn.softmax(tf.nn.bias_add(tf.matmul(X, W), b), name='relu_' + name)
    return X

def model(X, is_train=True):
    #VGG-16 Block1
    X = conv_forward(X, 64, (3,3), [1,1,1,1], 'SAME', name='block1_conv1')
    X = conv_forward(X, 64, (3,3), [1,1,1,1], 'SAME', name='block1_conv2')
    X = tf.nn.max_pool(X, [1,2,2,1], [1,2,2,1], 'VALID', name='block1_pool')
    
    #VGG-16 Block2
    X = conv_forward(X, 128, (3,3), [1,1,1,1], 'SAME', name='block2_conv1')
    X = conv_forward(X, 128, (3,3), [1,1,1,1], 'SAME', name='block2_conv2')
    X = tf.nn.max_pool(X, [1,2,2,1], [1,2,2,1], 'VALID', name='block2_pool')
    
    #VGG-16 Block3
    X = conv_forward(X, 256, (3,3), [1,1,1,1], 'SAME', name='block3_conv1')
    X = conv_forward(X, 256, (3,3), [1,1,1,1], 'SAME', name='block3_conv2')
    X = conv_forward(X, 256, (3,3), [1,1,1,1], 'SAME', name='block3_conv3')
    X = tf.nn.max_pool(X, [1,2,2,1], [1,2,2,1], 'VALID', name='block3_pool')
    
    #VGG-16 Block4
    X = conv_forward(X, 512, (3,3), [1,1,1,1], 'SAME', name='block4_conv1')
    X = conv_forward(X, 512, (3,3), [1,1,1,1], 'SAME', name='block4_conv2')
    X = conv_forward(X, 512, (3,3), [1,1,1,1], 'SAME', name='block4_conv3')
    X = tf.nn.max_pool(X, [1,2,2,1], [1,2,2,1], 'VALID', name='block4_pool')
    
    #VFF-16 Block5
    X = conv_forward(X, 512, (3,3), [1,1,1,1], 'SAME', name='block5_conv1')
    X = conv_forward(X, 512, (3,3), [1,1,1,1], 'SAME', name='block5_conv2')
    X = conv_forward(X, 512, (3,3), [1,1,1,1], 'SAME', name='block5_conv3')
    
    #Flatten X
    X = flatten(X)
    
    #Fully connected layers
    X = fc_forward(X, 4096, name='fc_1', activation='relu')
    X = fc_forward(X, 4096, name='fc_2', activation='relu')
    X = fc_forward(X, CONST_NUM_LABELS, name='preds', activation='softmax')
    return X

x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='X')
y = tf.placeholder(tf.float32, shape=[None, CONST_NUM_LABELS], name='labels')

with tf.name_scope('vgg-16-model'):
    logits = model(x)
    valid_logits = model(x, False)
    
with tf.name_scope('cost'):
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)
    
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(CONST_NUM_EPOCHS):
        for j in range(train_size//CONST_BATCH_SIZE):
            start = j*CONST_BATCH_SIZE
            end = j*CONST_BATCH_SIZE + CONST_BATCH_SIZE
            if end > train_size:
                end = train_size
            batch_data = X_train[start:end]
            batch_labels = T_train[start:end]
            #Batch execution
            _, preds, cost = sess.run([train, logits, cost], feed_dict={x:batch_data, y:batch_labels})
            if j%10 == 0:
                print('Epoch:' + str(i), 'Batch:' + str(j), 'Cost:' + str(cost))
            if j%100 == 0:
                preds = sess.run([valid_logits], feed_dict={x:X_valid, y:T_valid})
                p_valid = np.argmax(preds, axis=1)
                acc = np.mean(y_valid == p_valid)
                print('Epoch:' + str(i), 'Batch:' + str(j), 'Accuracy:' + str(acc))

    test_pred = np.zeros([test_size, CONST_NUM_LABELS])
    for k in range(test_size//CONST_BATCH_SIZE):
        start = k*CONST_BATCH_SIZE
        end = k*CONST_BATCH_SIZE + CONST_BATCH_SIZE
        if end > test_size:
            end = test_size
        batch_data = X_test[start:end]
        batch_labels = T_test[start:end]
        preds = sess.run([valid_logits], feed_dict={x:batch_data, y:batch_labels})
        test_pred[start:end] = preds
    
    p_test = np.argmax(test_pred, axis=1)
    test_acc = np.mean(y_test == p_test)
    print('Test Accuracy:' + str(test_acc))
    
    

    