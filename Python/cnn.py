# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:42:38 2018

@author: Erik
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import main as m

LEARNING_RATE = 0.05
DROPOUT_RATE = 0.5

TOPOLOGY_TYPE = 1

def cnn_model(features, labels, mode):
    inputLayer = tf.reshape(features["x"], [-1, 90, 90, 3])
    
    # MNIST-like Topology
    if (TOPOLOGY_TYPE == 1):
        conv1 = tf.layers.conv2d(
                inputs=inputLayer,
                filters=32,
                kernel_size=[7, 7],
                padding="valid",
                activation=tf.nn.relu)
        
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[5, 5],
                padding="valid",
                activation=tf.nn.relu)
        
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
        flatten = tf.reshape(pool2, [-1, 19 * 19 * 32])
        
        dense = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
        
        droplayer = tf.layers.dropout(
                inputs=dense,
                rate=DROPOUT_RATE,
                training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # AlexNET-like Topology    
    if (TOPOLOGY_TYPE == 2):
        conv1 = tf.layers.conv2d(
                inputs=inputLayer,
                filters=64,
                kernel_size=[7, 7],
                padding="valid",
                activation=tf.nn.relu)
        
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=1)
         
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="valid",
                activation=tf.nn.relu)
         
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
         
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
         
        conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=128,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
         
        conv5 = tf.layers.conv2d(
                inputs=conv4,
                filters=128,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
         
        pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
         
        flatten = tf.reshape(pool3, [-1, 16 * 16 * 128])
         
        dense1 = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.relu)
        
        droplayer1 = tf.layers.dropout(
                inputs=dense1,
                rate=DROPOUT_RATE,
                training=mode == tf.estimator.ModeKeys.TRAIN)
        
        dense2 = tf.layers.dense(inputs=droplayer1, units=2048, activation=tf.nn.relu)
        
        droplayer = tf.layers.dropout(
                inputs=dense2,
                rate=DROPOUT_RATE,
                training=mode == tf.estimator.ModeKeys.TRAIN)
    
     # VGGNET-like Topology    
    if (TOPOLOGY_TYPE == 3):
        conv1 = tf.layers.conv2d(
                inputs=inputLayer,
                filters=64,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
        
        conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
        
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
        conv3 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
        
        conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=128,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
        
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)
        
        conv5 = tf.layers.conv2d(
                inputs=pool2,
                filters=256,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
        
        conv6 = tf.layers.conv2d(
                inputs=conv5,
                filters=256,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu)
        
        pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[3, 3], strides=2)
        
        flatten = tf.reshape(pool3, [-1, 7 * 7 * 256])
        
        dense1 = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.relu)
        
        droplayer1 = tf.layers.dropout(
                inputs=dense1,
                rate=DROPOUT_RATE,
                training=mode == tf.estimator.ModeKeys.TRAIN)
        
        dense2 = tf.layers.dense(inputs=droplayer1, units=2048, activation=tf.nn.relu)
        
        droplayer = tf.layers.dropout(
                inputs=dense2,
                rate=DROPOUT_RATE,
                training=mode == tf.estimator.ModeKeys.TRAIN)         
         
    logits = tf.layers.dense(inputs=droplayer, units=m.CLASS_SIZE)
    
    predictions = {
     "classes": tf.argmax(input=logits, axis=1),
     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimiser.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)