#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:05:50 2022

@author: soyeonmin
"""

#Problem 4, 5 load cifar

import pickle
import os
import numpy as np

def unison_shuffled_copies(a, b, s):
    assert len(a) == len(b)
    np.random.seed(s)
    p = np.random.permutation(len(a))
    return a[p], b[p]

CIFAR_DOWNLOAD_PATH = '/Users/johnson/Desktop/S22_HW1_handout_v4/'
CIFAR_DOWNLOAD_PATH = os.path.join(CIFAR_DOWNLOAD_PATH, 'cifar-10-batches-py')

batch1 = pickle.load(open(os.path.join(CIFAR_DOWNLOAD_PATH, 'data_batch_1'), 'rb'), encoding='latin1')
batch2 = pickle.load(open(os.path.join(CIFAR_DOWNLOAD_PATH, 'data_batch_2'), 'rb'), encoding='latin1')
batch3 = pickle.load(open(os.path.join(CIFAR_DOWNLOAD_PATH, 'data_batch_3'), 'rb'), encoding='latin1')
batch4 = pickle.load(open(os.path.join(CIFAR_DOWNLOAD_PATH, 'data_batch_4'), 'rb'), encoding='latin1')
batch5 = pickle.load(open(os.path.join(CIFAR_DOWNLOAD_PATH, 'data_batch_5'), 'rb'), encoding='latin1')

test_batch = pickle.load(open(os.path.join(CIFAR_DOWNLOAD_PATH, 'test_batch'), 'rb'), encoding='latin1') 

trainX = np.concatenate([batch1['data'], batch2['data'], batch3['data'], batch4['data'], batch5['data']], axis=0) #shape is  (50000, 3072)
trainy = np.array(batch1['labels'] + batch2['labels'] + batch3['labels'] + batch4['labels'] +batch5['labels']) #shape is (50000,)

testX = test_batch['data']
testy = np.array(test_batch['labels'])

trainX, trainy = unison_shuffled_copies(trainX, trainy,0)
testX, testy = unison_shuffled_copies(testX, testy,1)

