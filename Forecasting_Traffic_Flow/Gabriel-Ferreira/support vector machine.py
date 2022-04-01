import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV


## Importing train and test data
train = pd.read_csv('/Users/gabrielvictorgomesferreira/Desktop/Analytics_Data_Science/train.csv')
test = pd.read_csv('/Users/gabrielvictorgomesferreira/Desktop/Analytics_Data_Science/test.csv')

## Puting time in the right format 
train['time'] = pd.to_datetime(train['time'], format = '%Y-%m-%d %H:%M:%S')
test['time'] = pd.to_datetime(test['time'], format = '%Y-%m-%d %H:%M:%S')

## Extracting day, hour and minute
train['day'] = train['time'].dt.dayofweek
train['hour'] = train['time'].dt.hour
train['minute'] = train['time'].dt.minute

test['day'] = test['time'].dt.dayofweek
test['hour'] = test['time'].dt.hour
test['minute'] = test['time'].dt.minute

## Changing direction to dummies
train = pd.concat([train, pd.get_dummies(train['direction'])], axis = 1)
test = pd.concat([test, pd.get_dummies(test['direction'])], axis = 1)

# Defining input and target variable
X = train.drop(['congestion', 'row_id', 'direction', 'time'], axis = 1)
Y = train['congestion']

# Splitting the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size = 0.2)

# Scaling the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_val = scaler.fit_transform(X_val)

## Defining the hyper-parameters for svm
svm_param_grid = {'kernel': ['rbf', 'poly', 'sigmoid'], 
                  'C': [0.01, 0.1, 1, 10],
                  'gamma': [0.001, 0.01, 0.1, 1]}


svm_grid_search = GridSearchCV(SVR(), svm_param_grid, cv = 3, scoring = 'neg_mean_squared_error', n_jobs = -1).fit(X_train, Y_train)

# Extracting the best model
svm_md = svm_grid_search.best_estimator_

# Predicting on validation and test
svm_val_pred = svm_md.predict(X_val)
svm_test_pred = svm_md.predict(X_test)
svm_test_dataset = svm_md.predict(test)

# Computing the mse on validation and test
svm_val_mse = mean_squared_error(Y_val, svm_val_pred)
svm_test_mse = mean_squared_error(Y_test, svm_test_pred)
svm_test_dataset_mse = mean_squared_error(Y_test, svm_test_pred)

mse_scores = pd.DataFrame({"svm_val_mse": svm_val_mse, 'svm_test_mse': svm_test_mse, "svm_test_dataset_mse": svm_test_dataset_mse})

svm_test_dataset.to_csv('svm_test_dataset.csv', index = False)
mse_scores.to_csv('mse_scores.csv', index = False)