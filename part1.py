# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import confusion_matrix

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

X_train, X_test, y_train, y_test = train_test_split(X, y)
# 'min_samples_leaf': list(range(10, 100,10))
# 'max_depth': list(range(10, 100,10))
# 'splitter' : ['best', 'random']
# 'criterion': ['gini', 'entropy']


# Grid Search CV with various Hyperparameters to determine best Model.
params = {'criterion': ['gini', 'entropy'], 'min_samples_leaf': list(range(10, 100,10)), 
          'max_depth': list(range(10, 100,10)),'splitter' : ['best', 'random']}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
grid_search_cv.fit(X_train, y_train)

print(grid_search_cv.best_estimator_)

best_model_task1 = grid_search_cv.best_estimator_

y_pred = best_model_task1.predict(X_test)

dot_data = StringIO()

export_graphviz(best_model_task1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names = best_model_task1.classes_)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png("bestmodeltree.png")

confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

## https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

# Accuracy
ac = (tp+tn)/(tn+tp+fp+fn)
# Misclassification Rate
mc = (fp+fn)/(tn+tp+fp+fn)
# True Positive Rate:
tpr = tp/(tp+fn)
# False Positive Rate:
fpr = fp/(fp+tn)
# True Negative Rate: 
tnr = tn/(fp+tn)
# Precision: 
pre = tp/(fp+tp)
# Prevalence: 
prev = (fn+tp)/(tn+tp+fp+fn)

print("Accuracy is " + str(ac*100) + "%")
print("Misclassification Rate is " + str(mc*100) + "%")
print("True Positive Rate is " + str(tpr*100) + "%")
print("False Positive Rate is " + str(fpr*100) + "%")
print("True Negative Rate " + str(tnr*100) + "%")
print("Precision is " + str(pre*100) + "%")
print("Prevalence is " + str(prev*100) + "%")