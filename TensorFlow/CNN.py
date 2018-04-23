#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:40:25 2018

@author: Das
"""

import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('../../MNIST_Data/train.csv', sep=',')

X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

X = np.array(X)
Y = np.array(Y)

#Normalize
X = X/255.0

X = X.astype(np.float32)
#Y = Y.astype(np.float32)

#X = X.reshape(X.shape[0], X.shape[1], 1)
X = X.reshape(X.shape[0], 28, 28, 1)

train_size = X.shape[0]

#some constants to declare
CONST_NUM_EPOCHS = 50
CONST_NUM_CHANNELS = 1
CONST_BATCH_SIZE = 64
CONST_IMAGE_SIZE = 28
CONST_NUM_LABELS = 10
CONST_SEED = 101

T = np.zeros((Y.size, CONST_NUM_LABELS))
T[np.arange(Y.size), Y] = 1

tf.set_random_seed(CONST_SEED)

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

def initialise_parameters():
    conv_w1 = tf.Variable(initial_value=tf.random_normal([5,5,
                    CONST_NUM_CHANNELS, 32], stddev=0.1), name='W')
    
    conv_b1 = tf.Variable(initial_value=tf.zeros(32), name='b')
    
    conv_w2 = tf.Variable(initial_value=tf.random_normal([5,5,32,64], 
                                            stddev=0.1), name='W')
    
    conv_b2 = tf.Variable(initial_value=tf.constant(0.1, shape=[64]), name='b')
    
    fc_w1 = tf.Variable(initial_value=tf.random_normal([CONST_IMAGE_SIZE//4 * 
                CONST_IMAGE_SIZE//4 * 64, 512], stddev = 0.1), name='W')
    
    fc_b1 = tf.Variable(initial_value=tf.constant(0.1, shape=[512]), name='b')
    
    fc_w2 = tf.Variable(initial_value=tf.random_normal([512, CONST_NUM_LABELS], 
                                               stddev=0.1), name='W')
    
    fc_b2 = tf.Variable(initial_value=tf.constant(0.1, shape=[CONST_NUM_LABELS]), name='b')
    
    return conv_w1, conv_b1, conv_w2, conv_b2, fc_w1, fc_b1, fc_w2, fc_b2

def conv_forward(X, W1, b1, W2, b2):
    conv1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
    pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv2 = tf.nn.conv2d(pool1, W2, strides=[1,1,1,1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
    pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    return pool2

def flatten_conv_inputs(conv):
    shape = tf.shape(conv)
    return tf.reshape(conv, [shape[0], shape[1]*shape[2]*shape[3]])
    

def fc_forward(X, W1, b1, W2, b2, train=True):
    hidden = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
    if train:
        hidden = tf.nn.dropout(hidden, keep_prob=0.5)
    output = tf.nn.softmax(tf.add(tf.matmul(hidden, W2), b2))
    return output

with tf.name_scope('init'):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w1, fc_b1, fc_w2, fc_b2 = initialise_parameters()

with tf.name_scope('conv'):
    conv_out = conv_forward(x, conv_w1, conv_b1, conv_w2, conv_b2)
    
with tf.name_scope('flatten'):
    flatten_out = flatten_conv_inputs(conv_out)
    
with tf.name_scope('fc'):
    logits = fc_forward(flatten_out, fc_w1, fc_b1, fc_w2, fc_b2)
    
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    
with tf.name_scope('train'):
    opt = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)
    
tf.summary.histogram("Conv_W1", conv_w1)
tf.summary.histogram("Conv_W2", conv_w2)
tf.summary.histogram("Conv_b1", conv_b1)
tf.summary.histogram("Conv_b2", conv_b2)
tf.summary.histogram("FC_W1", fc_w1)
tf.summary.histogram("FC_W2", fc_w2)
tf.summary.histogram("FC_b1", fc_b1)
tf.summary.histogram("FC_b2", fc_b2)
tf.summary.scalar("cost", cost)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/CNN_board/1', graph=tf.get_default_graph())
    
    sess.run(init)
    
    for step in range(CONST_NUM_EPOCHS*train_size // CONST_BATCH_SIZE):
        
        offset = (step * CONST_BATCH_SIZE) % (train_size - CONST_BATCH_SIZE)
        print(offset)
        batch_data = X[offset:(offset + CONST_BATCH_SIZE), ...]
        batch_labels = T[offset:(offset + CONST_BATCH_SIZE)]
        
        _, c, pred, summary = sess.run([opt, cost, logits, merged], feed_dict={x:batch_data, y:batch_labels})
        
        writer.add_summary(summary, step)
        
        if step%10 == 0:
            print("Epoch:", (step+1), "cost =", "{:.5f}".format(c))