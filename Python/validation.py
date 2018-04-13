# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:46:11 2018

@author: steph
"""

import numpy as np
import random
import main as m

def crossValidation(k):
    data, labels = m.getFrameMats('data/ucf-101/')
    label_data = list(zip(data, labels))
    random.shuffle(label_data)
    data, labels = list(zip(*label_data))
    accuracy = 0
    fold_size = len(data) // k
    
    basepath = "/Programs/Repos/INFOMCV_P4/Python/tmp/cnn_model"
    
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
        
        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        eval_data = np.asarray(eval_data)
        eval_labels = np.asarray(eval_labels)
        
        m.train(train_data, train_labels, model_path)
        results = m.predict(eval_data, eval_labels, model_path)
        
        print('Fold: ' + str(i))
    
    accuracy = accuracy / k
    
    return accuracy