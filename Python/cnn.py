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

LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

def cnn_model(features, labels, mode):
    inputLayer = tf.reshape(features["x"], [-1, 90, 90, 3])
    
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