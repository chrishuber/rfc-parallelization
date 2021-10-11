# Chris Huber
# SFSU 902339417
# Term Project
# 10/11/2021

# Purpose: 
# Parallelize Randomized Search CV and RandomForest Classifier using MPI-for-
# Python to reduce runtime for both, find optimal hyperparameters for rf_clf,
# and find maximum accuracy in minumum time using MNIST dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time
import math

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# import 70,000 MNIST records from OpenML
from sklearn.datasets import fetch_openml
mnist_data = fetch_openml('mnist_784', version=1, cache=True)

# extract data and labels
X = mnist_data.data
y = mnist_data.target

# divide data into train and test splits
X_train, X_test = train_test_split(X, test_size=0.2, random_state=25)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=25)


# Random search of parameters, using 5 fold cross validation, 
# search across 20 different combinations, and use n-1 cores
start_time = time.time()
rf_random = RandomizedSearchCV(estimator = rf_clf, param_distributions = random_grid, n_iter = 5, cv = 20, verbose=2, random_state=42, n_jobs = 7)# Fit the random search model
rf_random.fit(X_train, y_train)
end_time = time.time()
print("Elapsed time is {}".format(end_time-start_time))

# Output best params
rf_random.best_params_

# Fit RF model using best params <-- this should change
# rf_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=2, max_depth=60, bootstrap=False)
rf_clf.fit(X_train, y_train)

# Run Optimal RF_CLF
# start_time = time.time()
# cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy')
# end_time = time.time()
# print("Elapsed time is {}".format(end_time-start_time))

output score
# score = rf_clf.score(X_test, y_test)
# score



