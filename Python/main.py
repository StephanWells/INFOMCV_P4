# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:28:49 2018

@author: Erik
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

# Imports
import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import cnn
import validation as val

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 15
CLASS_SIZE = 5

class Action(Enum):
    BrushingTeeth = 0
    CuttingInKitchen = 1
    JumpingJack = 2
    Lunges = 3
    WallPushups = 4

def getFrameMat(path):
    vid = cv.VideoCapture(path)
    
    count = vid.get(cv.CAP_PROP_FRAME_COUNT)
    vid.set(cv.CAP_PROP_POS_FRAMES, count / 2)
    
    _, frame = vid.read()
    
    #cv.namedWindow('test', cv.WINDOW_AUTOSIZE)
    #cv.imshow('test', frame)
    #cv.waitKey(0)    
    #cv.destroyAllWindows()    
    #print(str(np.size(frame, 0)) + ',' + str(np.size(frame, 1)) + os.linesep)
    
    vid.release()
    return frame

def getFrameMatAt(path, pos):
    vid = cv.VideoCapture(path)
    
    vid.set(cv.CAP_PROP_POS_FRAMES, pos)
    
    _, frame = vid.read()
    
    vid.release()
    return frame

def getFrameMats(basepath):
    labels = []
    mats = []
    
    print("Video loading...", end='')
    for action in Action:
        datapath = basepath + action.name + '/'
        
        for filename in os.listdir(datapath):
            frame = getFrameMat(datapath + filename)
            mat = cv.resize(frame, (90, 90))
            
            mat_orig = np.divide(np.float32(mat), 255) 
            labels.append(action.value)
            mats.append(mat_orig)
            
            mat_flip = cv.flip(mat_orig, 1)            
            labels.append(action.value)
            mats.append(mat_flip)
            
            #cv.namedWindow('testorig', cv.WINDOW_AUTOSIZE)
            #cv.imshow('testorig', mat_orig)
            #cv.namedWindow('testflip', cv.WINDOW_AUTOSIZE)
            #cv.imshow('testflip', mat_flip)
            #cv.waitKey(0)
    
    print("DONE.")
    return mats, labels

def train(train_data, train_labels, model_path):
    classifier = tf.estimator.Estimator(model_fn=cnn.cnn_model, model_dir=model_path)
    
    log_tensors = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=log_tensors, every_n_iter=50)
    
    train_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=True)
    
    classifier.train(
            input_fn=train_input,
            steps=10000,
            hooks=[logging_hook])
    
def evaluate(eval_data, eval_labels, model_path):
    classifier = tf.estimator.Estimator(model_fn=cnn.cnn_model, model_dir=model_path)    

    eval_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    
    eval_results = classifier.evaluate(input_fn=eval_input)
    print(eval_results)
    
def predict(eval_data, eval_labels, model_path):
    classifier = tf.estimator.Estimator(model_fn=cnn.cnn_model, model_dir=model_path)
    
    eval_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            num_epochs=1,
            shuffle=False)
    
    eval_results = classifier.predict(input_fn=eval_input)
    
    return eval_results

def main():
#    ucf_mats, ucf_labels = getFrameMats('data/ucf-101/')
#    train_data = np.asarray(ucf_mats)
#    train_labels = np.asarray(ucf_labels)
#    
#    print(train_data)
#    print(train_labels)
#    
#    own_mats, own_labels = getFrameMats('data/own/')
#    eval_data = np.asarray(own_mats)
#    eval_labels = np.asarray(own_labels)
#    
#    print(eval_data)
#    print(eval_labels)
    val.crossValidation(5)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()