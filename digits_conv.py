import os
import numpy as np

def initialize_data():   
    path = 'trainingDigits/'
    file_list = os.listdir(path)
    m = len(file_list)
    n = 32
    X = np.zeros((m,n,n))
    for i in range(m):
        with open(path+file_list[i], 'r') as file:
            X_temp = file.readlines()
        for j in range(n):
            line = X_temp[j].strip()
            for k in range(n):
                X[i][j][k] = (int(line[k]))
    Y = np.zeros((m, 10))
    for i in range(m):
        digit = int(file_list[i].split('_')[0])
        Y[i][digit] = 1   
    return X, Y

def zero_padding(X, pad):
    return np.lib.pad(X, ((pad, pad), (pad, pad)), 
                      mode='constant', constant_values=0)

def conv_forward(A_prev, W, b, stride, pad):
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)
    n_H = int((n_H_prev-f+2*pad)/stride)+1
    n_W = int((n_W_prev-f+2*pad)/stride)+1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_padding(A_prev, pad)
