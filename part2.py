# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:59:39 2021

@author: rahul
"""

import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, recall_score, roc_auc_score, precision_score

ins_file = 'data'
ext = '.csv'

data = pd.read_csv(ins_file+ext)

# Data Wrangling
# Examining the actual dataset
print(data.head(20))

# Determine whether any null values are in the dataset
print(data.isnull().sum())
# No Null values exist
# How many unique values are in the data set per column
print(data.nunique())
# Determine the type of data of each column
print(data.dtypes)

# Determine the break up Yes an No values. this looks to be highly imbalanced.
# It will affect the results
print(data['Churn'].value_counts())

#Use ordinal Encoding to normalize the data. It also allows for a correlation
#heatmap analysis such that we can decide what columns to use as
#features

# Categorically Enconding each column with numbers to analyse the best features with a Heat Map. Also allows for better 
# accuracy in Tree Classifiers.
encoder = OrdinalEncoder()

data["gender_code"] = encoder.fit_transform(data.gender.values.reshape(-1, 1)).astype('int64')
data["Partner_code"] = encoder.fit_transform(data.Partner.values.reshape(-1, 1)).astype('int64')
data["Dependents_code"] = encoder.fit_transform(data.Dependents.values.reshape(-1, 1)).astype('int64')
data["PhoneService_code"] = encoder.fit_transform(data.PhoneService.values.reshape(-1, 1)).astype('int64')
data["MultipleLines_code"] = encoder.fit_transform(data.MultipleLines.values.reshape(-1, 1)).astype('int64')
data["InternetService_code"] = encoder.fit_transform(data.InternetService.values.reshape(-1, 1)).astype('int64')
data["OnlineSecurity_code"] = encoder.fit_transform(data.OnlineSecurity.values.reshape(-1, 1)).astype('int64')
data["OnlineBackup_code"] = encoder.fit_transform(data.OnlineBackup.values.reshape(-1, 1)).astype('int64')
data["DeviceProtection_code"] = encoder.fit_transform(data.DeviceProtection.values.reshape(-1, 1)).astype('int64')
data["TechSupport_code"] = encoder.fit_transform(data.TechSupport.values.reshape(-1, 1)).astype('int64')
data["StreamingTV_code"] = encoder.fit_transform(data.StreamingTV.values.reshape(-1, 1)).astype('int64')
data["StreamingMovies_code"] = encoder.fit_transform(data.StreamingMovies.values.reshape(-1, 1)).astype('int64')
data["Contract_code"] = encoder.fit_transform(data.Contract.values.reshape(-1, 1)).astype('int64')
data["PaperlessBilling_code"] = encoder.fit_transform(data.PaperlessBilling.values.reshape(-1, 1)).astype('int64')
data["PaymentMethod_code"] = encoder.fit_transform(data.PaymentMethod.values.reshape(-1, 1)).astype('int64')
data["Churn_code"] = encoder.fit_transform(data.Churn.values.reshape(-1, 1)).astype('int64')


# Make a heatmap on the correlations between variables in the hotel data:
data_correlations = data.corr()
data_correlations
plt.figure(figsize=(15,7))
fig = sns.heatmap(data_correlations, vmin = -1, vmax = 1, cmap = 'bwr', annot=True)
fig.figure.savefig("Correlation Heatmap.png")


# Features columns chosen for prediction.
feature_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Partner_code','Dependents_code','OnlineSecurity_code',
               'OnlineBackup_code', 'DeviceProtection_code','TechSupport_code','Contract_code','PaperlessBilling_code','PaymentMethod_code']

# Assigning Feature columns to X and y for Test Train Split
X = data[feature_cols]
y = data[['Churn']]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=99)

## Grid Search CV with various Hyperparameters to determine best Model.
#params = {'n_estimators': [100,150,200],'criterion': ['gini', 'entropy'], 'bootstrap': [True, False], 
#          'min_samples_leaf': [10, 20, 30, 40], 'max_depth': [10, 20, 30, 40]}
#
#grid_search_cv = GridSearchCV(RandomForestClassifier(), params, cv=5)
#grid_search_cv.fit(X_train, y_train)
#
#print(grid_search_cv.best_estimator_)

best_model_task2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=50, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=30, min_samples_split=5,
                       min_weight_fraction_leaf=0.03, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=90,
                       verbose=0, warm_start=False)

kf = model_selection.KFold(n_splits=10,shuffle=True)

best_model_task2.fit(X_train,y_train)
y_pred = best_model_task2.predict(X_test)

accuracy = cross_val_score(best_model_task2, X_test, y_test, cv=kf, scoring='accuracy')
roc_auc = cross_val_score(best_model_task2, X_test, y_test, cv=kf, scoring='roc_auc')

print(np.mean(accuracy)*100)
print(np.mean(roc_auc))
print(classification_report(y_test,y_pred))