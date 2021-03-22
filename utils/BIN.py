# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Evaluating Capability of Deep Neural Networks for Image Classification via Information Plane

NUM_LABEL = 10  # num of classes in training set
NUM_TEST_MASK = 128  # the number of samples in a batch
NUM_INTERVALS = 10  # the number of intervals of binning


def MI_cal(layer_T):
    '''
    Inputs:
    - size_of_test: (N,) how many test samples have be given. since every input is different
      we only care the number.
    -  label: the label of X.
    -  layer_T:  (N,H) H is the size of hidden layer
    Outputs:
    - MI_XT : the mutual information I(X,T)
    - MI_TY : the mutual information I(T,Y)
    '''
    MI_XT = 0
    layer_T = Discretize(layer_T)
    XT_matrix = np.zeros((NUM_TEST_MASK, NUM_TEST_MASK))
    Non_repeat = []
    mark_list = []
    for i in range(NUM_TEST_MASK):
        pre_mark_size = len(mark_list)
        if i == 0:
            Non_repeat.append(i)
            mark_list.append(i)
            XT_matrix[i, i] = 1
        else:
            for j in range(len(Non_repeat)):
                if (layer_T[i] == layer_T[Non_repeat[j]]).all():
                    mark_list.append(Non_repeat[j])
                    XT_matrix[i, Non_repeat[j]] = 1
                    break
        if pre_mark_size == len(mark_list):
            Non_repeat.append(Non_repeat[-1] + 1)
            mark_list.append(Non_repeat[-1])
            XT_matrix[i, Non_repeat[-1]] = 1

    XT_matrix = np.delete(XT_matrix, range(len(Non_repeat), NUM_TEST_MASK), axis=1)
    P_layer_T = np.sum(XT_matrix, axis=0) / float(NUM_TEST_MASK)
    P_sample_x = 1 / float(NUM_TEST_MASK)
    for i in range(NUM_TEST_MASK):
        MI_XT += P_sample_x * np.log2(1.0 / P_layer_T[mark_list[i]])

    return MI_XT


def Discretize(layer_T):
    '''
    Discretize the output of the neuron
    Inputs:
    - layer_T:(N,H)
    Outputs:
    - layer_T:(N,H) the new layer_T after discretized
    '''
    labels = np.arange(NUM_INTERVALS)
    pos_list = np.arange(NUM_INTERVALS / 2 + 1) * (1.0 / (NUM_INTERVALS / 2))
    neg_list = -pos_list
    neg_list.sort()
    bins = np.append(neg_list, pos_list)
    bins = np.delete(bins, int(NUM_INTERVALS / 2))
    for i in range(layer_T.shape[1]):
        temp = pd.cut(layer_T[:, i], bins, labels=labels)
        layer_T[:, i] = np.array(temp)
    return layer_T

if __name__ == '__main__':
    a = np.random.rand(128, 20)

    print(MI_cal(a))