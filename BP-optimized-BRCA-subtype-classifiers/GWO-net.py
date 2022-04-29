from statistics import mode
import pandas as pd
import numpy  as np
import math
import csv
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
import GWO
import tensorflow as tf
import time



# Control the scale of the searching space
min_values = -5
max_values = 5

global dimension # dimensionality of input
dimension = 0
global bias
bias = []
global weights
weights = []
global model
model = Sequential()

# Function to be Minimized.
def target_function (variables_values = np.zeros(dimension)):
    print("weight dim: ", np.array(variables_values).shape)
    # variables_values = variables_values.tolist()

    # Initialize weights and biases
    num_node_weight = 0
    num_node_bias = 0
    for i in range(len(model.layers)):
        num_node_weight = num_node_weight + len(model.layers[i].get_weights()[0])
        num_node_bias = num_node_bias + len(model.layers[i].get_weights()[1])

    bias = np.random.rand(num_node_bias)
    bias = bias.tolist()

    total_wei = 0
    total_bia = 0
    for i in range(len(model.layers)):
        weights_temp_len = len(model.layers[i].get_weights()[0])
        weights_layer_len = len(model.layers[i].get_weights()[0][0])
        weights_temp = []
        for j in range(weights_temp_len):
            weights_layer_temp = variables_values[total_wei : total_wei + weights_layer_len]
            weights_temp.append(weights_layer_temp)
            total_wei = total_wei + weights_layer_len

        bias_temp_len = len(model.layers[i].get_weights()[1])
        bias_temp = bias[total_bia : total_bia + bias_temp_len]
        total_bia = total_bia + bias_temp_len

        model.layers[i].set_weights([np.array(weights_temp), np.array(bias_temp)])

    model.fit(X_train, y_train, epochs=1, batch_size=64)

    y_pred = model.predict(X_val)
    seec = tf.keras.losses.SparseCategoricalCrossentropy()
    return seec(y_val,y_pred).numpy()
    

f1 = []
accuracy_list=[]
time_ls=[]
for k in range(10):
    # Data preprocessing
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    with open("./10cv_datasets/group_"+str(k+1)+"_train_X.csv",'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_train.append(row)

    with open("./10cv_datasets/group_"+str(k+1)+"_train_Y.csv",'r') as f:
        reader = csv.reader(f)
        for row in reader:
            y_train.append(row)

    with open("./10cv_datasets/group_"+str(k+1)+"_test_X.csv",'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_val.append(row)

    with open("./10cv_datasets/group_"+str(k+1)+"_test_Y.csv",'r') as f:
        reader = csv.reader(f)
        for row in reader:
            y_val.append(row)

    X_train.remove(X_train[0])
    X_val.remove(X_val[0])
    y_train.remove(y_train[0])
    y_val.remove(y_val[0])

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            X_train[i][j] = float(X_train[i][j])
    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            y_train[i][j] = int(y_train[i][j]) - 1
    for i in range(len(X_val)):
        for j in range(len(X_val[i])):
            X_val[i][j] = float(X_val[i][j])
    for i in range(len(y_val)):
        for j in range(len(y_val[i])):
            y_val[i][j] = int(y_val[i][j]) - 1

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Neural network
    model = Sequential()
    model.add(Dense(125, input_dim=1000, activation='relu'))
    
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Initialize weights and biases
    num_node_weight = 0
    num_node_bias = 0
    for i in range(len(model.layers)):
        num_node_weight = num_node_weight + len(model.layers[i].get_weights()[0]) * len(model.layers[i].get_weights()[1])
        num_node_bias = num_node_bias + len(model.layers[i].get_weights()[1])
    
    dimension = num_node_weight

    bias = np.random.rand(num_node_bias)
    bias = bias.tolist()
    weights = np.random.rand(num_node_weight)
    weights = weights.tolist()

    
    #important: iteration = 50
    start_t=time.time()
    gwo = GWO.GWO(target_function, 0, 1, dimension, 10, 50)
    time_ls.append(time.time()-start_t)
    weights = gwo.Alpha_pos
    
    total_wei = 0
    total_bia = 0
    for i in range(len(model.layers)):
        weights_temp_len = len(model.layers[i].get_weights()[0])
        weights_layer_len = len(model.layers[i].get_weights()[0][0])
        weights_temp = []
        for j in range(weights_temp_len):
            weights_layer_temp = weights[total_wei : total_wei + weights_layer_len]
            weights_temp.append(weights_layer_temp)
            total_wei = total_wei + weights_layer_len

        bias_temp_len = len(model.layers[i].get_weights()[1])
        bias_temp = bias[total_bia : total_bia + bias_temp_len]
        total_bia = total_bia + bias_temp_len

        model.layers[i].set_weights([np.array(weights_temp), np.array(bias_temp)])

#important: epochs =100
    history = model.fit(X_train, y_train, epochs=100, batch_size=64)

    y_pred = model.predict(X_val)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))

    f1.append(f1_score(y_val, pred, average='weighted'))

    y_val = np.reshape(y_val, -1)
    #print(y_val)
    #print(np.sum(y_val_l == predd), y_val_l.shape, predd.shape)
    acc_temp = np.sum(y_val == pred) / y_val.shape[0]
    accuracy_list.append(acc_temp)


print("f1:", f1)
print("ACC: ", accuracy_list)
s = 0
for i in range(len(f1)):
    s = s + f1[i]
print("f1 average:", s/len(f1))
print("time duration:", time_ls)
a = 0
for i in range(len(time_ls)):
    a = a + time_ls[i]
print("average time duration:", a/len(time_ls))