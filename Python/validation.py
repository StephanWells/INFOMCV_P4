# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:46:11 2018

@author: steph
"""

import numpy as np, cv2
import random
import main as m

def crossValidation(k):
    data, labels = m.getFrameMats('data/ucf-101/')
    label_data = list(zip(data, labels))
    random.shuffle(label_data)
    data, labels = list(zip(*label_data))
    accuracy = 0
    fold_size = len(data) // k
    
    for i in range(0, k):
        end_offset = 1 if i < (k - 1) else 0
        left_data = data[0:fold_size * i]
        left_labels = labels[0:fold_size * i]
        right_data = data[fold_size * (i + 1) + end_offset:]
        right_labels = labels[fold_size * (i + 1) + end_offset:]
        
        train_data = []
        train_labels = []
        train_data.extend(left_data)
        train_data.extend(right_data)
        train_labels.extend(left_labels)
        train_labels.extend(right_labels)
        
        eval_data = []
        eval_labels = []
        eval_data.extend(data[fold_size * i:fold_size * (i + 1)])
        eval_labels.extend(labels[fold_size * i:fold_size * (i + 1)])
        
        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        eval_data = np.asarray(eval_data)
        eval_labels = np.asarray(eval_labels)
        
        m.classify(train_data, train_labels, eval_data, eval_labels)
        
        print('Fold: ' + i)
    
    accuracy = accuracy / k
    
    return accuracy

def generateConfusionMatrix(predicted_labels, actual_labels):
    return

def normaliseConfusionMatrix(conf_mat):
    for i in range(0, conf_mat.shape[0]):
        temp_total = 0;
        
        for j in range(0, conf_mat.shape[1]):
            temp_total = temp_total + conf_mat[i, j]
        
        for j in range(0, conf_mat.shape[1]):
            conf_mat[i, j] = np.float32(conf_mat[i, j]) / np.float32(temp_total)
    
    return conf_mat

def outputConfusionMatrix(conf_mat):
    if conf_mat.shape[0] != conf_mat.shape[1]:
        print('ERROR: Confusion matrix is not a square matrix')
        
        return
    
    tile_size = 100
    conf_mat = normaliseConfusionMatrix(conf_mat)
    conf_output = np.zeros((tile_size * conf_mat.shape[0], tile_size * conf_mat.shape[0], 3))
    
    for i in range(0, conf_mat.shape[0]):
        for j in range(0, conf_mat.shape[1]):
            conf_val = conf_mat[i, j]
            colour = np.array([conf_val, 0, 0])
            
            for k in range(0, tile_size):
                for l in range(0, tile_size):
                    conf_output[i * tile_size + k, j * tile_size + l] = colour
            
            text_val = "{:.2f}".format(conf_val)
            text_location = (j * tile_size + (tile_size // 3), i * tile_size + (tile_size // 2))
            text_font = cv2.FONT_HERSHEY_PLAIN
            text_scale = 1
            text_colour = (255, 255, 255)
            text_thickness = 1
            text_line = cv2.LINE_AA
            
            cv2.putText(conf_output, text_val, text_location, text_font, text_scale, text_colour, text_thickness, text_line)
            
    cv2.imshow('Confusion Matrix', conf_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()