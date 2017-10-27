#!/usr/bin/python3

import os
import numpy as np

def initialize_data(path):
    X_temp = []  
    file_list = os.listdir(path)
    m = len(file_list)   
    for train_file in file_list:
        with open(path+train_file,'r') as train_x:
            X_temp.extend(train_x.readlines())   
    X = []   
    for i in X_temp:
        i = i.strip()
        for j in i:
            X.append(int(j))    
    X = np.array(X).reshape(m,-1).T   
    Y_temp = []    
    for i in file_list:
        Y_temp.append(int(i.split('_')[0]))    
    Y_temp = np.array(Y_temp).T
    Y = np.zeros((10, m))
    for i in range(m):
        Y[Y_temp[i],i] = 1    
    return X, Y

def initialize_param(layer_dims):
    parameters = {}
    layers = len(layer_dims)
    for l in range(1,layers):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], 
                   layer_dims[l-1])*np.sqrt(2.0/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def forward_prop(train_X,parameters):
    cache = {}
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    w4 = parameters['w4']
    b4 = parameters['b4']
    
    z1 = np.dot(w1, train_X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = np.tanh(z3)
    z4 = np.dot(w4, a3) + b4
    a4 = sigmoid(z4)
    
    cache['a1'] = a1
    cache['a2'] = a2
    cache['a3'] = a3
    cache['a4'] = a4
    return a4, cache

def back_prop(train_X,parameters,cache,train_Y,m):
    a1 = cache['a1']
    a2 = cache['a2']
    a3 = cache['a3']
    a4 = cache['a4']
    w4 = parameters['w4']
    w3 = parameters['w3']
    w2 = parameters['w2']
    
    dz4 = a4 - train_Y
    dw4 = 1.0/m*np.dot(dz4, a3.T)
    db4 = 1.0/m*np.sum(dz4, axis=1, keepdims=True)
    
    dz3 = np.dot(w4.T, dz4)*(1-a3**2)
    dw3 = 1.0/m*np.dot(dz3, a2.T)
    db3 = 1.0/m*np.sum(dz3, axis=1, keepdims=True)
    
    dz2 = np.dot(w3.T, dz3)*(1-a2**2)
    dw2 = 1.0/m*np.dot(dz2, a1.T)
    db2 = 1.0/m*np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = np.dot(w2.T, dz2)*(1-a1**2)
    dw1 = 1.0/m*np.dot(dz1, train_X.T)
    db1 = 1.0/m*np.sum(dw1, axis=1, keepdims=True)
    
    gradients = {}
    gradients['dw4'] = dw4
    gradients['db4'] = db4
    gradients['dw3'] = dw3
    gradients['db3'] = db3
    gradients['dw2'] = dw2
    gradients['db2'] = db2
    gradients['dw1'] = dw1
    gradients['db1'] = db1
    
    return gradients
    
def train(iter_nums, train_X, train_Y, layer_dims, learning_rate):
    m = train_X.shape[1]
    parameters = initialize_param(layer_dims)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    w4 = parameters['w4']
    b4 = parameters['b4']
    Y_predict,cache = forward_prop(train_X, parameters)

    for i in range(iter_nums):
        a3, cache = forward_prop(train_X, parameters)
        gradients = back_prop(train_X, parameters, cache, train_Y,m)
        dw1 = gradients['dw1']
        db1 = gradients['db1']
        dw2 = gradients['dw2']
        db2 = gradients['db2']
        dw3 = gradients['dw3']
        db3 = gradients['db3']
        dw4 = gradients['dw4']
        db4 = gradients['db4']
        w1 = w1 - learning_rate*dw1
        b1 = b1 - learning_rate*db1
        w2 = w2 - learning_rate*dw2
        b2 = b2 - learning_rate*db2
        w3 = w3 - learning_rate*dw3
        b3 = b3 - learning_rate*db3
        w4 = w4 - learning_rate*dw4
        b4 = b4 - learning_rate*db4
        parameters['w1'] = w1
        parameters['b1'] = b1
        parameters['w2'] = w2
        parameters['b2'] = b2
        parameters['w3'] = w3
        parameters['b3'] = b3
        parameters['w4'] = w4
        parameters['b4'] = b4
        Y_predict, cache = forward_prop(train_X, parameters)
        loss = compute_loss(Y_predict, train_Y)
        if i%50==0:
            print("iter_nums = " + str(i) +", loss is: " + str(loss))
    return parameters
    
def predict(X, parameters):
    Y_predict, cache = forward_prop(X, parameters)
    m = Y_predict.shape[1]
    for i in range(m):
        Y_predict[:,i] = 1*(Y_predict[:,i]>=Y_predict[:,i].max())
    return restore(Y_predict)

def comp():
    train_X,train_Y = initialize_data('trainingDigits/')
    test_X, test_Y = initialize_data('testDigits/')
    layer_dims = [1024, 100, 50, 30, 10]
    parameters = train(10000,train_X,train_Y,layer_dims,0.02)
    train_accuracy = compute_accuracy(parameters, train_X, restore(train_Y))
    test_accuracy = compute_accuracy(parameters, test_X, restore(test_Y))
    return train_accuracy, test_accuracy

def compute_accuracy(parameters, X, Y):
    m = Y.shape[1]
    Y_predict = predict(X, parameters)
    count = 0
    for i in range(m):
        if Y_predict[0,i] == Y[0,i]:
            count += 1
    accuracy = count/m
    return accuracy
    

def restore(Y):
    m = Y.shape[1]
    Y_restore = np.zeros((1,m))
    for i in range(m):
        Y_restore[0,i] = (np.where(Y[:,i]==1))[0]
    return Y_restore

def compute_loss(a, Y):
    loss = np.sum(-1*Y*np.log(a) - (1-Y)*np.log(1-a))
    return loss
    
def sigmoid(a):
    return (1/(1+pow(np.e,-1*a)))