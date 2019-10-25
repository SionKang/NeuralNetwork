#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:25:10 2019

@author: sionkang
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def layers(X, Y):
    input_size = X.shape[-1]
    hidden_size1 = 32
    hidden_size2 = 32
    output_size = Y.shape[-1]
    return input_size, hidden_size1, hidden_size2, output_size
#input_size is # of features
#hidden_size is # of nodes in 1 hiddle layer = 4
#output_size = 1
    
def sigmoid(Z):
    
   s = 1/(1+np.exp(-Z))
   
   return s

def tanh(Z):
    
    th = np.divide((np.exp(Z) - np.exp(-Z)), (np.exp(Z) + np.exp(-Z)))
    
    return th

def reLU(Z):
    
    Zr = Z
    Zr[Zr<0] = 0
    r = Zr
    
    return r

def deriv_reLU(Z):
    
    Zr = Z
    Zr[Zr<0] = 0
    Zr[Zr>=0] = 1
    deriv_r = Zr
    
    return deriv_r
        

def initialization(input_size, hidden_size1, hidden_size2, output_size):
    W1 = np.random.randn(input_size,hidden_size1)*0.01
    b1 = np.zeros((1,hidden_size1))
    W2 = np.random.randn(hidden_size1,hidden_size2)*0.01
    b2 = np.zeros((1,hidden_size2))
    W3 = np.random.randn(hidden_size2,output_size)*0.01
    b3 = np.zeros((1,output_size))

    
    parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3
    }
    
    return parameters
#we want different theta values for each nodes
#we intialize theta as random values close to 0
#find function that generates random values in right dimension (numpy)
#theta0/b = 0

def forward_propagation(X, parameters):
    #implement forward propagation
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(X,W1)+b1
    A1 = reLU(Z1)
    Z2 = np.dot(A1,W2)+b2
    A2 = reLU(Z2)
    Z3 = np.dot(A2,W3)+b3
    A3 = sigmoid(Z3)
    
    return Z1,A1,Z2,A2,Z3,A3
#A1 = tanh(Z1)
#A2 = sigmoid(Z2)
#Z is linear equation formed with W and b

def cost_computation(A3, Y):
    m = Y.shape[0]
    
    cost = Y*np.log(A3) + (1-Y)*np.log(1-A3)
    cost = sum(cost)
    cost = cost*(-1/m)
    
    return cost
#cost function

def backward_propagation(parameters, Z1, A1, Z2, A2, Z3, A3, X, Y):
    dZ3 = A3 - Y
    dW3 = (1/X.shape[0])*np.dot(A2.T,dZ3)
    db3 = (1/X.shape[0])*np.sum(dZ3)
    deriv_r2 = deriv_reLU(Z2)
    dZ2 = np.dot(dZ3,parameters["W3"].T)*deriv_r2
    dW2 = (1/X.shape[0])*np.dot(A1.T,dZ2)
    db2 = (1/X.shape[0])*np.sum(dZ2)
    deriv_r1 = deriv_reLU(Z1)
    dZ1 = np.dot(dZ2,parameters["W2"].T)*deriv_r1
    dW1 = (1/X.shape[0])*np.dot(X.T,dZ1)
    db1 = (1/X.shape[0])*np.sum(dZ1)
    
    gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
            "dW3": dW3,
            "db3": db3,
            
    }
    
    return gradients
#partial derivatives and update parameters
#refer to page 4

def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] = parameters["W1"] - gradients["dW1"] * learning_rate
    parameters["b1"] = parameters["b1"] - gradients["db1"] * learning_rate
    parameters["W2"] = parameters["W2"] - gradients["dW2"] * learning_rate
    parameters["b2"] = parameters["b2"] - gradients["db2"] * learning_rate
    parameters["W3"] = parameters["W3"] - gradients["dW3"] * learning_rate
    parameters["b3"] = parameters["b3"] - gradients["db3"] * learning_rate
    
    return parameters
#using values from backward_propogation, update parameters

def nn_model(X, Y, hidden_size1, hidden_size2, num_iters, alpha):
    input_size, hidden_size1, hidden_size2, output_size = layers(X, Y)
    parameters = initialization(input_size, hidden_size1, hidden_size2, output_size)
    cost = np.zeros(num_iters)
    
    for i in range(num_iters):
        Z1,A1,Z2,A2,Z3,A3 = forward_propagation(X, parameters)
        cost[i] = cost_computation(A3, Y)
        gradients = backward_propagation(parameters, Z1, A1, Z2, A2, Z2, A3, X, Y)
        parameters = update_parameters(parameters, gradients, alpha)
        
    return parameters, cost
#use all functions above to find finalized optimal parameters

def accuracy(parameters, X, Y):
    Z1,A1,Z2,A2,Z3,A3 = forward_propagation(X, parameters)
    
    boolean = (A3 >= 0.5).astype(int)
    
    boolean = boolean.flatten()

    accurate = np.mean(boolean == Y)
    
    return accurate*100
#forward propogation, find predicted y values, comparison with actual y values
    
def evaluateData(data, split, num_iters, alpha):   
#    y = data.PE.values
#    y =  data["eval"].values
#    meany = np.mean(y)
#    boolean = (y >= meany).astype(int)
#    boolean = boolean.flatten()
#    y = boolean
#    X = data[data_features[:4]]
#    X = data[data_features[:11]]
    
    data = data.values
    y = data[:, -1]
    X = data[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
#    X_train = X_train.values
#    X_test = X_test.values
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]
    
    meanTrain = np.ones(X_train.shape[1])
    stdTrain = np.ones(X_train.shape[1])
    meanTest = np.ones(X_test.shape[1])
    stdTest = np.ones(X_test.shape[1])
    
    for i in range(X_train.shape[1]):
        meanTrain[i] = np.mean(X_train.T[i])
        stdTrain[i] = np.std(X_train.T[i])
            
    Normalized_X_train = np.ones(X_train.shape)
    for j in range(X_train.shape[1]):
        Normalized_X_train[:,j] = (X_train.T[j] - meanTrain[j])/stdTrain[j]
        
    for i in range(X_test.shape[1]):
        meanTest[i] = np.mean(X_test.T[i])
        stdTest[i] = np.std(X_test.T[i])
        
    Normalized_X_test = np.ones(X_test.shape)
    for j in range(X_test.shape[1]):
        Normalized_X_test[:,j] = (X_test.T[j] - meanTest[j])/stdTest[j]
        
    input_size, hidden_size1, hidden_size2, output_size = layers(Normalized_X_train, y_train)
    
    parameters, cost = nn_model(Normalized_X_train, y_train, hidden_size1, hidden_size2, num_iters, alpha)
    
    accurateTrain = accuracy(parameters, Normalized_X_train, y_train)
    accurateTest = accuracy(parameters, Normalized_X_test, y_test)
    
    fig, ax = plt.subplots()  
    ax.plot(np.arange(iters), cost, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')  
    ax.set_title('Error vs. Training Epoch')  

    Jtrain = cost[-1]
    accuracyTrain = accurateTrain
    accuracyTest = accurateTest

    return Jtrain, accuracyTrain, accuracyTest

#TESTCLASS
#data_features = ['AT','V','AP','RH','PE']
#data = pd.read_csv('electricityNEW.csv')
data = pd.read_csv("housepriceNN.csv")

iters = 100000
alpha = 0.05
split = 0.3
#
Jtrain, accuracyTrain, accuracyTest = evaluateData(data, split, iters, alpha)

print()
print("Train Data:")
print("Cost Value: " + str(round(np.asscalar(Jtrain),2)))
print("Accuracy: " + str(round(np.asscalar(accuracyTrain),2)) + "%")
print()
print("Test Data:")
print("Accuracy: " + str(round(np.asscalar(accuracyTest),2)) + "%")









