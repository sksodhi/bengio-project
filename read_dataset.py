#!/usr/bin/python

import numpy as np

#
# This function reads letter recognition dataset
# Dataset location: http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
#
#Calling this function
#
#X,T = read_letter_recognition_dataset('./datasets/letter-recognition/letter-recognition.data')
#
def read_letter_recognition_dataset(data_file_name):
    num_rows=20000
    num_cols=16
    X = np.empty((num_rows, num_cols))
    T = []
    r = 0
    data_file = open(data_file_name, 'r')
    for line in data_file:
        c = 0
        for field in line.strip().split(','):
            if c == 0:
                T.append(field)
            else:
                x_col = c - 1
                X[r][x_col] = field
            c += 1
        r += 1
            #print (field)

    return X,T



