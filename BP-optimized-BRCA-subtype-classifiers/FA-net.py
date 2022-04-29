from statistics import mode
import pandas as pd
import numpy as np
import math
import csv
import os
import datetime
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, balanced_accuracy_score


min_values = -5
max_values = 5

global dimension  
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



def initial_fireflies(swarm_size=10, dim=dimension):
    position = pd.DataFrame(np.zeros((swarm_size, dim)))
    position['Fitness'] = 0.0
    temp = weights
    temp.append(0.0)
    p = pd.DataFrame(np.array(temp))
    for i in range(0, swarm_size):
        position.iloc[i] = p[0]
        position.iloc[i, -
                      1] = target_function(position.iloc[i, 0:position.shape[1]-1])
    return position

# Function: Distance Calculations


def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance
    return distance**(1/2)

# Function: Beta Value


def beta_value(x, y, gama=1, beta_0=1):
    rij = euclidean_distance(x, y)
    beta = beta_0*math.exp(-gama*(rij)**2)
    return beta

# Function: Ligth Intensity


def ligth_value(light_0, x, y, gama=1):
    rij = euclidean_distance(x, y)
    light = light_0*math.exp(-gama*(rij)**2)
    return light

# Function: Update Position


def update_position(position, x, y, alpha_0=0.2, beta_0=1, gama=1, firefly=1):
    updated_position = position.copy(deep=True)
    for j in range(0, len(x)):
        epson = int.from_bytes(os.urandom(
            8), byteorder="big") / ((1 << 64) - 1) - (1/2)
        updated_position.iloc[i, j] = (
            x[j] + beta_value(x, y, gama=gama, beta_0=beta_0)*(y[j] - x[j]) + alpha_0*epson)
        if (updated_position.iloc[i, j] > max_values):
            updated_position.iloc[i, j] = max_values
        elif (updated_position.iloc[i, j] < min_values):
            updated_position.iloc[i, j] = min_values
    updated_position.iloc[firefly, -1] = target_function(
        updated_position.iloc[firefly, 0:position.shape[1]-1])
    return updated_position

# Function: Initialize Destination Position


def destination_position(dimension=dimension):
    destination = pd.DataFrame(np.zeros((1, dimension)))
    destination['Fitness'] = 0.0
    destination.iloc[0, -
                     1] = target_function(destination.iloc[0, 0:destination.shape[1]-1])
    return destination

# Function: Update Destination by Fitness


def update_destination(position, destination):
    updated_position = position.copy(deep=True)
    for i in range(0, position.shape[0]):
        if (updated_position.iloc[i, -1] < destination.iloc[0, -1]):
            destination.iloc[0] = updated_position.iloc[i]
    return destination

# FA Function


def firefly_algorithm(swarm_size=10, dim=dimension, generations=50, alpha_0=0.2, beta_0=1, gama=1):
    count = 1
    position = initial_fireflies(swarm_size=swarm_size, dim=dim)
    best_firefly = destination_position(dimension=dim)
    while (count <= generations):
        print(count)
        for i in range(0, swarm_size):
            for j in range(0, swarm_size):
                if (i != j):
                    firefly_i = np.array(position[i])
                    firefly_j = np.array(position[j])
                    ligth_i = ligth_value(
                        position[i][1], firefly_i, firefly_j, gama=gama)
                    ligth_j = ligth_value(
                        position[j][1], firefly_i, firefly_j, gama=gama)
                    
                    if (ligth_i > ligth_j):
                        position = update_position(
                            position, firefly_i, firefly_j, alpha_0=alpha_0, beta_0=beta_0, gama=gama, firefly=i)
        best_firefly = update_destination(position, best_firefly)
        count = count + 1
    
    return best_firefly.iloc[best_firefly['Fitness'].idxmin()]

# Function to be Minimized.


def target_function(variables_values=np.zeros(dimension)):
    variables_values = variables_values.tolist()

    # Initialize weights and biases
    num_node_weight = 0
    num_node_bias = 0
    for i in range(len(model.layers)):
        num_node_weight = num_node_weight + \
            len(model.layers[i].get_weights()[0])
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
            weights_layer_temp = variables_values[total_wei: total_wei +
                                                  weights_layer_len]
            weights_temp.append(weights_layer_temp)
            total_wei = total_wei + weights_layer_len

        bias_temp_len = len(model.layers[i].get_weights()[1])
        bias_temp = bias[total_bia: total_bia + bias_temp_len]
        total_bia = total_bia + bias_temp_len

        model.layers[i].set_weights(
            [np.array(weights_temp), np.array(bias_temp)])

    model.fit(X_train, y_train, epochs=50, batch_size=64)

    y_pred = model.predict(X_val)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()

    return scce(y_val, y_pred).numpy()


f1 = []
accuracy = []

for k in range(10):
    # Data preprocessing
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    with open("./10cv_datasets/group_"+str(k+1)+"_train_X.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_train.append(row)

    with open("./10cv_datasets/group_"+str(k+1)+"_train_Y.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            y_train.append(row)

    with open("./10cv_datasets/group_"+str(k+1)+"_test_X.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_val.append(row)

    with open("./10cv_datasets/group_"+str(k+1)+"_test_Y.csv", 'r') as f:
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
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Initialize weights and biases
    num_node_weight = 0
    num_node_bias = 0
    for i in range(len(model.layers)):
        num_node_weight = num_node_weight + \
            len(model.layers[i].get_weights()[0]) * \
            len(model.layers[i].get_weights()[1])
        num_node_bias = num_node_bias + len(model.layers[i].get_weights()[1])

    dimension = num_node_weight

    bias = np.random.rand(num_node_bias)
    bias = bias.tolist()
    weights = np.random.rand(num_node_weight)
    weights = weights.tolist()

    print('Start time: {0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
    weights = firefly_algorithm(
        swarm_size=1, dim=dimension, generations=50, alpha_0=0.2, beta_0=1, gama=1)
    print('End time: {0}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))

    total_wei = 0
    total_bia = 0
    for i in range(len(model.layers)):
        weights_temp_len = len(model.layers[i].get_weights()[0])
        weights_layer_len = len(model.layers[i].get_weights()[0][0])
        weights_temp = []
        for j in range(weights_temp_len):
            weights_layer_temp = weights[total_wei: total_wei +
                                         weights_layer_len]
            weights_temp.append(weights_layer_temp)
            total_wei = total_wei + weights_layer_len

        bias_temp_len = len(model.layers[i].get_weights()[1])
        bias_temp = bias[total_bia: total_bia + bias_temp_len]
        total_bia = total_bia + bias_temp_len

        model.layers[i].set_weights(
            [np.array(weights_temp), np.array(bias_temp)])

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
