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

tf.logging.set_verbosity(tf.logging.INFO)

class Action(Enum):
    BrushingTeeth = 1
    CuttingInKitchen = 2
    JumpingJack = 3
    Lunges = 4
    WallPushups = 5

class LabelMat:
    label = None
    mat = None

def getFrameMat(path):
    vid = cv.VideoCapture(path)
    
    count = vid.get(cv.CAP_PROP_FRAME_COUNT)
    vid.set(cv.CAP_PROP_POS_FRAMES, count / 2)
    
    _, frame = vid.read()
    
    cv.namedWindow('test', cv.WINDOW_AUTOSIZE)
    cv.imshow('test', frame)
    cv.waitKey(0)
    
    vid.release()
    cv.destroyAllWindows()
    return frame

def getFrameMats():
    basepath = 'data/ucf-101/'
    frames = []
    
    for action in Action:
        datapath = basepath + action.name + '/'
        
        for filename in os.listdir(datapath):
            lm = LabelMat()
            lm.label = action
            
            frame = getFrameMat(datapath + filename)
            lm.mat = cv.resize(frame, (90, 90))
            
            print(lm.mat.size)
            
            frames.append(lm)
    
    return frames

def main():
    getFrameMats()
    
    
if __name__ == "__main__":
    main()