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
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.05, shuffle=True)

#One-hot vectorizing
T_train = np.zeros([y_train.size, 6])
T_train[np.arange(y_train.size), y_train] = 1

T_valid = np.zeros([y_valid.size, 6])
T_valid[np.arange(y_valid.size), y_valid] = 1

T_test = np.zeros([y_test.size, 6])
T_test[np.arange(y_test.size), y_test] = 1

train_size = X_train.shape[0]
test_size = X_test.shape[0]

#get weights
def get_filters(channel_in, channel_out, kernel):
    shape = [kernel[0], kernel[1], channel_in, channel_out]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, seed=CONST_SEED))

#get bias
def get_bias(channel_out):
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[channel_out]))

#convolution
def conv_forward(X, channel_out, kernel, strides, padding, name):
    shape = X.get_shape().as_list()
    channel_in = shape[-1]
    filtr = get_filters(channel_in, channel_out, kernel)
    return tf.nn.conv2d(X, filtr, strides, padding, name=name)

#batch_normalization
def batch_normalization(X, channel_out, name, is_conv=True):
    beta = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[channel_out]), 
                       trainable=True)
    gamma = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[channel_out]), 
                        trainable=True)
    
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
    norm = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3, name=name)
    return norm

def activation_forward(X, name):
    shape = X.get_shape().as_list()
    channel_out = shape[-1]
    bias = get_bias(channel_out)
    return tf.nn.relu(tf.nn.bias_add(X, bias), name=name)

def identity_block(X, kernel, channels_out, stage, block):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    CO1, CO2, CO3 = channels_out
    
    X_shortcut = X
    
    #First component
    X = conv_forward(X, channel_out=CO1, kernel=(1,1), strides=[1,1,1,1], padding='VALID', name=conv_name_base + '2a')
    X = batch_normalization(X, channel_out=CO1, name=bn_name_base + '2a')
    X = activation_forward(X, name='relu')
    
    #Second component
    X = conv_forward(X, channel_out=CO2, kernel=kernel, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2b')
    X = batch_normalization(X, channel_out=CO2, name=bn_name_base + '2b')
    X = activation_forward(X, name='relu')
    
    X = conv_forward(X, channel_out=CO3, kernel=(1,1), strides=[1,1,1,1], padding='VALID', name=conv_name_base + '2c')
    X = batch_normalization(X, channel_out=CO3, name=bn_name_base + '2c')
    
    X = tf.add(X, X_shortcut)
    X = activation_forward(X, 'relu')
    
    return X

def conv_block(X, kernel, channels_out, stage, block, s):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    CO1, CO2, CO3 = channels_out
    
    X_shortcut = X
    
    #First component
    X = conv_forward(X, channel_out=CO1, kernel=(1,1), strides=[1,s,s,1], padding='VALID', name = conv_name_base + '2a')
    X = batch_normalization(X, channel_out=CO1, name=bn_name_base + '2a')
    X = activation_forward(X, name='relu')
    
    #Second component
    X = conv_forward(X, channel_out=CO2, kernel=kernel, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2b')
    X = batch_normalization(X, channel_out=CO2, name=bn_name_base + '2b')
    X = activation_forward(X, name='relu')
    
    #Third component
    X = conv_forward(X, channel_out=CO3, kernel=(1,1), strides=[1,1,1,1], padding='VALID', name=conv_name_base + '2c')
    X = batch_normalization(X, channel_out=CO3, name=bn_name_base + '2c')
    
    #shortcut = Ws * shortcut
    X_shortcut = conv_forward(X_shortcut, channel_out=CO3, kernel=(1,1), strides=[1,s,s,1], padding='VALID', name=conv_name_base + '1')
    X_shortcut = batch_normalization(X_shortcut, channel_out=CO3, name=bn_name_base + '1')
    
    X = tf.add(X, X_shortcut)
    X = activation_forward(X, name='relu')
    return X

def flatten(X):
    shape = X.get_shape().as_list()
    return tf.reshape(X, [tf.shape(X)[0], shape[1]*shape[2]*shape[3]])

def fc_forward(X, channel_out):
    shape = X.get_shape().as_list()
    channel_in = shape[-1]
    W = tf.Variable(tf.truncated_normal([channel_in,channel_out], stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1), name='b')
    X = tf.nn.relu(tf.add(tf.matmul(X, W), b), name='relu')
    return X

def ResNetModel(X):
    
    #Stage 1
    X = conv_forward(X, 64, (7,7), [1,2,2,1], padding='SAME', name='conv1')
    X = batch_normalization(X, 64, name='bn_conv1')
    X = activation_forward(X, 'relu')
    X = tf.nn.max_pool(X, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
    #pooling_forward(X, kernel=(3,3), strides=[1,2,2,1], padding='VALID')
    
    #Stage 2
    X = conv_block(X, kernel=(3,3), channels_out=(64, 64, 256), stage=2, block='a', s=1)
    X = identity_block(X, kernel=(3,3), channels_out=(64, 64, 256), stage=2, block='b')
    X = identity_block(X, kernel=(3,3), channels_out=(64, 64, 256), stage=2, block='c')
    
    #Stage 3
    X = conv_block(X, kernel=(3,3), channels_out=(128, 128, 512), stage=3, block='a', s=2)
    X = identity_block(X, kernel=(3,3), channels_out=(128,128,512), stage=3, block='b')
    X = identity_block(X, kernel=(3,3), channels_out=(128,128,512), stage=3, block='c')
    X = identity_block(X, kernel=(3,3), channels_out=(128,128,512), stage=3, block='d')

    #Stage 4    
    X = conv_block(X, kernel=(3,3), channels_out=(256, 256, 1024), stage=4, block='a', s=2)
    X = identity_block(X, kernel=(3,3), channels_out=(256, 256, 1024), stage=4, block='b')
    X = identity_block(X, kernel=(3,3), channels_out=(256, 256, 1024), stage=4, block='c')
    X = identity_block(X, kernel=(3,3), channels_out=(256, 256, 1024), stage=4, block='d')
    X = identity_block(X, kernel=(3,3), channels_out=(256, 256, 1024), stage=4, block='e')    
    X = identity_block(X, kernel=(3,3), channels_out=(256, 256, 1024), stage=4, block='f')
    
    #Stage 5
    X = conv_block(X, kernel=(3,3), channels_out=(512, 512, 2048), stage=5, block='a', s=2)
    X = identity_block(X, kernel=(3,3), channels_out=(512, 512, 2048), stage=5, block='b')
    X = identity_block(X, kernel=(3,3), channels_out=(512, 512, 2048), stage=5, block='c')
    
    #AveragePool
    X = tf.nn.avg_pool(X, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')
    
    #Flatten
    X = flatten(X)
    
    #softmax
    X = fc_forward(X, CONST_NUM_CLASSES)
    X = tf.nn.softmax(X)
    return X


#TF placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='X')
y = tf.placeholder(dtype=tf.int32, shape=[None, CONST_NUM_CLASSES], name='labels')
is_train = tf.placeholder(dtype=tf.bool, name='training_phase')

with tf.name_scope('ResNet50'):
    logits = ResNetModel(x)
    
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

with tf.name_scope('opt'):
    opt = tf.train.AdamOptimizer().minimize(cost)

tf.summary.scalar('cost', cost)
merged = tf.summary.merge_all()
    
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    writer = tf.summary.FileWriter("logs/ResNet_board/1", graph=tf.get_default_graph())
    sess.run(init)
    
    for i in range(1):
        for j in range(train_size//CONST_BATCH_SIZE):
            start = j*CONST_BATCH_SIZE
            end = j*CONST_BATCH_SIZE + CONST_BATCH_SIZE
            
            batch_input = X_train[start:end, ...]
            batch_labels = T_train[start:end, ...]
            
            _, c, oput, summary = sess.run([opt, cost, logits, merged], feed_dict={x:batch_input, y:batch_labels, is_train:True})
            writer.add_summary(summary, i)
            
            valid_cost, valid_logits = sess.run([cost, logits], feed_dict={x:X_valid, y:T_valid, is_train:False})
    