#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:40:25 2018

@author: Das
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


#some constants
CONST_NUM_EPOCHS = 2
CONST_NUM_CHANNELS = 1
CONST_BATCH_SIZE = 256
CONST_IMAGE_SIZE = 28
CONST_NUM_LABELS = 10
CONST_SEED = 101


'''df = pd.read_csv('../../MNIST_Data/train.csv', sep=',')

X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

X = np.array(X)
Y = np.array(Y)

#Normalize
X = X/255.0

X = X.astype(np.float32)
#Y = Y.astype(np.float32)

#X = X.reshape(X.shape[0], X.shape[1], 1)
X = X.reshape(X.shape[0], 28, 28, 1)'''

mnist = tf.contrib.learn.datasets.load_dataset('mnist')
X_train = mnist.train.images
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
X_test = mnist.test.images
y_test = np.asarray(mnist.test.labels, dtype=np.int32)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.05, shuffle=True)

train_size = X_train.shape[0]
test_size = X_test.shape[0]

T_train = np.zeros((y_train.size, CONST_NUM_LABELS))
T_train[np.arange(y_train.size), y_train] = 1

T_valid = np.zeros((y_valid.size, CONST_NUM_LABELS))
T_valid[np.arange(y_valid.size), y_valid] = 1

T_test = np.zeros((y_test.size, CONST_NUM_LABELS))
T_test[np.arange(y_test.size), y_test] = 1

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

def variable_summaries(name, var):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var,mean))))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram',var)


with tf.name_scope('init'):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w1, fc_b1, fc_w2, fc_b2 = initialise_parameters()

with tf.name_scope('conv'):
    conv_out = conv_forward(x, conv_w1, conv_b1, conv_w2, conv_b2)
    
with tf.name_scope('flatten'):
    flatten_out = flatten_conv_inputs(conv_out)
    
with tf.name_scope('fc'):
    logits = fc_forward(flatten_out, fc_w1, fc_b1, fc_w2, fc_b2)
    valid_logits = fc_forward(flatten_out, fc_w1, fc_b1, fc_w2, fc_b2, False)
    
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

with tf.name_scope('valid_cost'):
    valid_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=valid_logits))
    
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)
    
scalar_labels = tf.argmax(y, axis=1)
scalar_logits = tf.argmax(logits, axis=1)
scalar_valid_logits = tf.argmax(valid_logits, axis=1)

with tf.name_scope('acc'):
    acc = 1 - tf.reduce_mean(tf.cast(tf.equal(scalar_labels, scalar_logits), tf.float32))

with tf.name_scope('valid_acc'):
    valid_acc = 1 - tf.reduce_mean(tf.cast(tf.equal(scalar_labels, scalar_valid_logits), tf.float32))
    
variable_summaries("Conv_W1", conv_w1)
variable_summaries("Conv_W2", conv_w2)
tf.summary.histogram("Conv_b1", conv_b1)
tf.summary.histogram("Conv_b2", conv_b2)
variable_summaries("FC_W1", fc_w1)
variable_summaries("FC_W2", fc_w2)
tf.summary.histogram("FC_b1", fc_b1)
tf.summary.histogram("FC_b2", fc_b2)
tf.summary.scalar("cost", cost)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('logs/CNN_board/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter('logs/CNN_board/test', graph=tf.get_default_graph())
    
    sess.run(init)
    
    for step in range(CONST_NUM_EPOCHS*train_size // CONST_BATCH_SIZE):
        
        offset = (step * CONST_BATCH_SIZE) % (train_size - CONST_BATCH_SIZE)
        batch_data = X_train[offset:(offset + CONST_BATCH_SIZE), ...]
        batch_labels = T_train[offset:(offset + CONST_BATCH_SIZE)]
        
        _, c, pred, accuracy, summary = sess.run([train, cost, logits, acc, merged], feed_dict={x:batch_data, y:batch_labels})
        
        if step%100 == 0:
            v_c, v_pred, v_acc, v_summary = sess.run([valid_cost, valid_logits, valid_acc, merged], feed_dict={x:X_valid, y:T_valid})
            train_writer.add_summary(summary, step)
            test_writer.add_summary(v_summary, step)
            #print("Epoch:", (step+1), "cost =", "{:.5f}".format(c), "acc =", "{:.5f}".format(accuracy))
            print("Epoch:", (step+1), "cost =", "{:.5f}".format(c), 
                  "v_cost =", "{:.5f}".format(v_c), "acc =", "{:.5f}".format(accuracy), 
                  "v_acc =", "{:.5f}".format(v_acc))
    
    test_pred = np.ndarray([test_size, CONST_NUM_LABELS])
    for step in (0, test_size, CONST_BATCH_SIZE):
        if step * CONST_BATCH_SIZE + CONST_BATCH_SIZE <= test_size:
            start = step * CONST_BATCH_SIZE
            end = step * CONST_BATCH_SIZE+CONST_BATCH_SIZE
        else:
            start = step * CONST_BATCH_SIZE
            end = test_size
        
        batch_data = X_test[start:end]
        batch_labels = T_test[start:end]
        
        t_pred = sess.run([valid_logits], feed_dict={x:batch_data, y:batch_labels})
        test_pred[start:end] = t_pred
        
predictions = np.argmax(test_pred, axis=1)
test_acc = 1 - np.mean(y_test == predictions)
        
        