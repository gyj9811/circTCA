# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:00:55 2020

@author: guoyajing
"""

import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow
import random
from sklearn.model_selection import train_test_split
from keras.layers import Concatenate
import scipy
from scipy import stats
sigma = 1.0
mu = 0.0
P_array = np.zeros(101)
norm = np.zeros(24)

def gauss():
    for i in range(24):
        norm[i] = scipy.stats.norm(mu,sigma).pdf(i)*100
    for i in range(24):
        P_array[50+i] = norm[i]
        P_array[50-i] = norm[i]
    return P_array  #101*1

coden_dict1 = {'A':0,
              'U':1,
              'C':2,
              'G':3,
              }

def coden(seq):
    vectors = np.zeros((len(seq), 4))
    for i in range(len(seq)):
            vectors[i][coden_dict1[seq[i].replace('T', 'U')]] = 1 + P_array[i]   #101*4
    return vectors.tolist()

def coden1(seq):
    vectors = np.zeros((len(seq), 4))
    for i in range(len(seq)):
            vectors[i][coden_dict1[seq[i].replace('T', 'U')]] = 1   #101*4
    return vectors.tolist()

coden_dict2 = {'AAA':0,'AAU':1,'AAC':2,'AAG':3,
              'AUA':4,'AUU':5,'AUC':6,'AUG':7,
              'ACA':8,'ACU':9,'ACC':10,'ACG':11,
              'AGA':12,'AGU':13,'AGC':14,'AGG':15,
              'UAA':16,'UAU':17,'UAC':18,'UAG':19,
              'UUA':20,'UUU':21,'UUC':22,'UUG':23,
              'UCA':24,'UCU':25,'UCC':26,'UCG':27,
              'UGA':28,'UGU':29,'UGC':30,'UGG':31,
              'CAA':32,'CAU':33,'CAC':34,'CAG':35,
              'CUA':36,'CUU':37,'CUC':38,'CUG':39,
              'CCA':40,'CCU':41,'CCC':42,'CCG':43,
              'CGA':44,'CGU':45,'CGC':46,'CGG':47,
              'GAA':48,'GAU':49,'GAC':50,'GAG':51,
              'GUA':52,'GUU':53,'GUC':54,'GUG':55,
              'GCA':56,'GCU':57,'GCC':58,'GCG':59,
              'GGA':60,'GGU':61,'GGC':62,'GGG':63,
              }

def coden2(seq):
    vectors2 = np.zeros((len(seq), 64))
    seq = seq + seq[0] +seq[1]
    for i in range(len(seq) - 2):
        vectors2[i][coden_dict2[seq[i:i+3].replace('T', 'U')]] = 1
    return vectors2.tolist()

def dealwithdata(protein):
    # 数据准备
    protein = protein
    dataX1 = []
    dataX2 = []
    dataY = []
    with open('gdatasetunblaced_1/' + protein + '/halfhalfpositive') as f:
        for line in f:
            line = line.strip('\n')
            if '>' not in line:
                dataX1.append(coden(line.strip()))
                dataX2.append(coden2(line.strip()))
                dataY.append([0, 1])
    with open('gdatasetunblaced_1/' + protein + '/negative') as f:
        for line in f:
            line = line.strip('\n')
            if '>' not in line:
                dataX1.append(coden(line.strip()))
                dataX2.append(coden2(line.strip()))
                dataY.append([1, 0])
    #不可以取相同数字
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX1 = np.array(dataX1)[indexes]
    dataX2 = np.array(dataX2)[indexes]
    dataY = np.array(dataY)[indexes]
    train_X1, test_X1, train_y, test_y = train_test_split(dataX1, dataY, test_size=0.2, random_state=0)
    train_X2, test_X2, train_y, test_y =  train_test_split(dataX2, dataY, test_size=0.2, random_state=0)
    return train_X1, train_X2, test_X1,test_X2, train_y, test_y
