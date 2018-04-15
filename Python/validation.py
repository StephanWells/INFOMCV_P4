# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:46:11 2018

@author: steph
"""

import numpy as np
import cv2 as cv
import random
import main as m
import os
    
basepath = "/Programs/Repos/INFOMCV_P4/Python/tmp/cnn_model"

def crossValidation(k):
    data, labels = m.getFrameMats('data/ucf-101/')
    label_data = list(zip(data, labels))
    random.shuffle(label_data)
    data, labels = list(zip(*label_data))
    fold_size = len(data) // k
    
    overall_conf_mat = np.zeros((m.CLASS_SIZE, m.CLASS_SIZE, 1))
    
    for i in range(0, k):
        model_path = basepath + "_fold" + str(i)
        
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
        
        train_data_arr = np.asarray(train_data)
        train_labels_arr = np.asarray(train_labels)
        eval_data_arr = np.asarray(eval_data)
        eval_labels_arr = np.asarray(eval_labels)
        
        m.train(train_data_arr, train_labels_arr, model_path)
        results = m.predict(eval_data_arr, eval_labels_arr, model_path)
        
        predicted_labels = []
        
        for result in results:
            predicted_labels.append(result['classes'])
            
        conf_mat = generateConfusionMatrix(eval_labels, predicted_labels)
        overall_conf_mat = np.add(overall_conf_mat, conf_mat)       
        
        print('Fold: ' + str(i))
        
    averaged_conf_mat = np.divide(np.float32(overall_conf_mat), k)
    outputConfusionMatrix(averaged_conf_mat) 
    generatePerfMeasures(averaged_conf_mat)

def predictSingleVideo(path, frameNum=-1):
    model_path = basepath + "_fold0"
    
    eval_data = []
    eval_labels = []
    
    if frameNum == -1:
        eval_data.append(m.getFrameMatWithResize(path))
    else:
        eval_data.append(m.getFrameMatWithResizeAt(path, frameNum))
        
    eval_data_arr = np.asarray(eval_data)
    eval_labels_arr = np.asarray(eval_labels)
    
    results = m.predict(eval_data_arr, eval_labels_arr, model_path)
    
    for result in results:
        label = m.Action(result['classes']).name
        print('Predicted Class: ' + str(label))

def testSingleFrame(k):
    overall_conf_mat = np.zeros((m.CLASS_SIZE, m.CLASS_SIZE, 1))
    
    for i in range(0, k):    
        model_path = basepath + "_fold" + str(i)
        
        eval_data, eval_labels = m.getFrameMats('data/own/')
        eval_data_arr = np.asarray(eval_data)
        eval_labels_arr = np.asarray(eval_labels)
        
        results = m.predict(eval_data_arr, eval_labels_arr, model_path)
        
        predicted_labels = []
        
        for result in results:
            predicted_labels.append(result['classes'])
            
        conf_mat = generateConfusionMatrix(eval_labels, predicted_labels)
        overall_conf_mat = np.add(overall_conf_mat, conf_mat)
    
    averaged_conf_mat = np.divide(np.float32(overall_conf_mat), k)
    outputConfusionMatrix(averaged_conf_mat) 
    generatePerfMeasures(averaged_conf_mat)
    
def testAllFrame(k):
    overall_conf_mat = np.zeros((m.CLASS_SIZE, m.CLASS_SIZE, 1))
    
    for i in range(0, k):    
        model_path = basepath + "_fold" + str(i)
        
        eval_data, eval_labels = m.getFrameMatsAll('data/own/')
        eval_data_arr = np.asarray(eval_data)
        eval_labels_arr = np.asarray(eval_labels)
        
        results = m.predict(eval_data_arr, eval_labels_arr, model_path)
        
        predicted_labels = []
        
        for result in results:
            predicted_labels.append(result['classes'])
            
        conf_mat = generateConfusionMatrix(eval_labels, predicted_labels)
        overall_conf_mat = np.add(overall_conf_mat, conf_mat)
    
    averaged_conf_mat = np.divide(np.float32(overall_conf_mat), k)
    outputConfusionMatrix(averaged_conf_mat) 
    generatePerfMeasures(averaged_conf_mat)   

def generatePerfMeasures(conf_mat):
    outFile = open("scores.txt", "w")
    
    for i in range(0, m.CLASS_SIZE):
        tp = conf_mat[i, i]
        
        tpfp = 0
        for j in range(0,5):
            tpfp += conf_mat[i, j]
        
        tpfn = 0
        for j in range(0,5):
            tpfn += conf_mat[j, i]
            
        precision = tp / tpfp
        recall = tp / tpfn
        fscore = (2 * precision * recall) / (precision + recall)
        
        outFile.write("Precision: " + str(precision) + ";")
        outFile.write("Recall: " + str(recall) + ";")
        outFile.write("F-Score: " + str(fscore) + os.linesep)
    
    outFile.close()

def generateConfusionMatrix(actual_labels, predicted_labels):
    conf_mat = np.zeros((m.CLASS_SIZE, m.CLASS_SIZE, 1))
    
    for i in range(0, len(actual_labels)):
            conf_mat[actual_labels[i], predicted_labels[i]] += 1
    
    return conf_mat

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
            
            text_val = str(round(float(conf_val), 2))
            text_location = (j * tile_size + (tile_size // 3), i * tile_size + (tile_size // 2))
            text_font = cv.FONT_HERSHEY_PLAIN
            text_scale = 1
            text_colour = (255, 255, 255)
            text_thickness = 1
            text_line = cv.LINE_AA
            
            cv.putText(conf_output, text_val, text_location, text_font, text_scale, text_colour, text_thickness, text_line)
            
    cv.imshow('Confusion Matrix', conf_output)