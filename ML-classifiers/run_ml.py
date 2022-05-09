import sys
import pandas as pd
import numpy as np
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree, svm, linear_model, neighbors
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

training_x_fileName = sys.argv[1]
training_y_fileName = sys.argv[2]
testing_x_fileName = sys.argv[3]
testing_y_fileName = sys.argv[4]
ver_name = sys.argv[5]

x_data = pd.read_csv(training_x_fileName, index_col = False)
y_data = pd.read_csv(training_y_fileName, index_col = False)
x_test = pd.read_csv(testing_x_fileName, index_col = False)
y_test = pd.read_csv(testing_y_fileName, index_col = False)

y_data = np.ravel(y_data)
y_test = np.ravel(y_test)

svm_classifier = svm.SVC(kernel = 'rbf', probability=True) #, random_state=0
svm_classifier = svm_classifier.fit(x_data, y_data)
svm_predicted_class = svm_classifier.predict(x_test)
acc = accuracy_score(y_test, svm_predicted_class)
svm_scores = svm_classifier.predict_proba(x_test)[:,1]


resultFile = "prediction_group_" + ver_name + "_SVM.csv"
f = open(resultFile, 'w')
for i in range(len(svm_predicted_class)) :
        data = "%s\n" % svm_predicted_class[i]
        #data = "%.0f\n" % svm_predicted_class[i]
        f.write(data)
f.close()

rf_classifier = RandomForestClassifier() #random_state=0
rf_classifier = rf_classifier.fit(x_data, y_data)
rf_predicted_class = rf_classifier.predict(x_test)
acc = accuracy_score(y_test, rf_predicted_class)
rf_scores = rf_classifier.predict_proba(x_test)[:,1]


resultFile = "prediction_group_" + ver_name +  "_RF.csv"
f = open(resultFile, 'w')
for i in range(len(rf_predicted_class)) :
        #data = "%.0f\n" % rf_predicted_class[i]
        data = "%s\n" % rf_predicted_class[i]
        f.write(data)
f.close()

dt_classifier = tree.DecisionTreeClassifier()
dt_classifier = dt_classifier.fit(x_data, y_data)
dt_predicted_class = dt_classifier.predict(x_test)
acc = accuracy_score(y_test, dt_predicted_class)
dt_scores = dt_classifier.predict_proba(x_test)[:,1]

resultFile = "prediction_group_" + ver_name + "_DT.csv"
f = open(resultFile, 'w')
for i in range(len(dt_predicted_class)) :
        #data = "%.0f\n" % rf_predicted_class[i]
        data = "%s\n" % dt_predicted_class[i]
        f.write(data)
f.close()

naivebayes_classifier = GaussianNB()
naivebayes_classifier = naivebayes_classifier.fit(x_data, y_data)
naivebayesn_predicted_class = naivebayes_classifier.predict(x_test)
acc = accuracy_score(y_test, naivebayesn_predicted_class)
nb_scores = naivebayes_classifier.predict_proba(x_test)[:,1]

resultFile = "prediction_group_" + ver_name + "_NB.csv"
f = open(resultFile, 'w')
for i in range(len(naivebayesn_predicted_class)) :
        #data = "%.0f\n" % rf_predicted_class[i]
        data = "%s\n" % naivebayesn_predicted_class[i]
        f.write(data)
f.close()

lr_classifier = LogisticRegression() #random_state=0
lr_classifier = lr_classifier.fit(x_data, y_data)
lr_predicted_class = lr_classifier.predict(x_test)
acc = accuracy_score(y_test, lr_predicted_class)
lr_scores = lr_classifier.predict_proba(x_test)[:,1]

resultFile = "prediction_group_" + ver_name +  "_LR.csv"
f = open(resultFile, 'w')
for i in range(len(lr_predicted_class)) :
        #data = "%.0f\n" % rf_predicted_class[i]
        data = "%s\n" % lr_predicted_class[i]
        f.write(data)
f.close()
