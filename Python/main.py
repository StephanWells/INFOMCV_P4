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

tf.logging.set_verbosity(tf.logging.INFO)

class Action(Enum):
    BrushingTeeth = 'BrushingTeeth'
    CuttingInKitchen = 'CuttingInKitchen'
    JumpingJack = 'JumpingJack'
    Lunges = 'Lunges'
    WallPushups = 'WallPushups'    

class LabelMat:
    label = None
    mat = None
    tensor = None

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

def main():
    getFrameMat('data/ucf-101/BrushingTeeth/v_BrushingTeeth_g01_c01.avi')
    
    
if __name__ == "__main__":
    main()