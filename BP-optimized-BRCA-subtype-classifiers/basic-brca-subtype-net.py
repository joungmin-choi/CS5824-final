import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score

# Data preprocessing
f1 = []

for k in range(10):
    # Neural network
    model = Sequential()
    model.add(Dense(500, input_dim=1000, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

    history = model.fit(X_train, y_train, epochs=100, batch_size=64)

    y_pred = model.predict(X_val)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))

    f1.append(f1_score(y_val, pred, average='weighted'))

print(f1)
s = 0
for i in range(len(f1)):
    s = s + f1[i]
print(s/len(f1))