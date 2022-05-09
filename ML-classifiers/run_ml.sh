#!/bin/bash

dirname="./10cv_datasets/"
for i in {1..10}
do
 x_data=$dirname"group_"$i"_train_X.csv"
 y_data=$dirname"group_"$i"_train_Y.csv"
 x_test=$dirname"group_"$i"_test_X.csv"
 y_test=$dirname"group_"$i"_test_Y.csv"

 python3 run_ml.py $x_data $y_data $x_test $y_test $i
done

