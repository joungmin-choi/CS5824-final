import time
from statistics import mode
import pandas as pd
import numpy  as np
import math
import csv
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, balanced_accuracy_score
import ALO

# Sine Cosine Algorithm

# Control the scale of the searching space
min_values = -5
max_values = 5

global dimension # dimensionality of SCA input
dimension = 0
global bias
bias = []
global weights
weights = []
global model
model = Sequential()
global X_train
X_train = []
global X_val
X_val = []
global y_train
y_train = []
global y_val
y_val = []

# Function to be Minimized.
def target_function (variables_values = np.zeros(dimension)):
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
    scce = tf.keras.losses.SparseCategoricalCrossentropy()

    return scce(y_val, y_pred).numpy()


f1 = []
accuracy = []

for k in range(1):
    print("***********************************")
    print("Now fold: ", k+1)
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
    alo = ALO.ALO(N=10, Max_iter=50, lb=np.zeros(dimension), ub=np.ones(dimension), dim=dimension, Fobj=target_function)
    start=time.time()
    _, weights, _ = alo.Run()
    print("--- optimization time is %s seconds ---" % (time.time() - start))

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

    history = model.fit(X_train, y_train, epochs=100, batch_size=64)

    y_pred = model.predict(X_val)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))

    f1.append(f1_score(y_val, pred, average='weighted'))
    accuracy.append(balanced_accuracy_score(y_val, pred))

print("F1-score:")
print(f1)
s = 0
for i in range(len(f1)):
    s = s + f1[i]
print(s/len(f1))

print("Accuracy:")
print(accuracy)
s = 0
for i in range(len(accuracy)):
    s = s + accuracy[i]
print(s/len(accuracy))
