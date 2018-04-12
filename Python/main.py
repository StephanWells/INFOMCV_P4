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

tf.logging.set_verbosity(tf.logging.INFO)

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
    
    #print(str(np.size(frame, 0)) + ',' + str(np.size(frame, 1)) + os.linesep)
    vid.release()
    #cv.destroyAllWindows()
    return frame

def getFrameMats():
    basepath = 'data/ucf-101/'
    labels = []
    mats = []
    
    print("Video loading...", end='')
    for action in Action:
        datapath = basepath + action.name + '/'
        
        for filename in os.listdir(datapath):
            labels.append(action.value)
            
            frame = getFrameMat(datapath + filename)
            mat = cv.resize(frame, (90, 90))           
            mats.append(np.float32(mat))
    
    print("DONE.")
    return mats, labels

def main():
    frame_mats, frame_labels = getFrameMats()
    train_data = np.asarray(frame_mats)
    train_labels = np.asarray(frame_labels)
    
    classifier = tf.estimator.Estimator(model_fn=cnn.cnn_model, model_dir="/tmp/cnn_model")
    
    log_tensors = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=log_tensors, every_n_iter=50)
    
    train_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=10,
            num_epochs=None,
            shuffle=True)
    
    classifier.train(
            input_fn=train_input,
            steps=20000,
            hooks=[logging_hook])    
    
if __name__ == "__main__":
    main()