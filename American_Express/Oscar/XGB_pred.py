import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from xgboost import XGBRegressor
from Amex_Metric import amex_metric

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'AmericanExpress/Delinquency_Features_Filled.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading data-files
data = pd.read_csv(file_content_stream, usecols = ['D_44_median', 'D_44_mean', 'D_44_max', 
                                                   'D_75_max', 'D_75_mean', 'D_78_max', 
                                                   'D_78_mean', 'D_78_range', 'D_44_std', 
                                                   'D_75_median', 'D_78_std', 'D_74_mean',
                                                   'D_44_range', 'D_44_min', 'D_84_mean', 
                                                   'D_74_max', 'D_41_range', 'D_75_min', 
                                                   'D_44_IQR', 'D_84_range', 'target'])

## Defining input and target 
X = data.drop(columns = 'target', axis = 1)
Y = data['target']

## Spliting the data into train, validation, and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y)


############
## Optuna ##
############

def objective_amex(trial):
    
    ## Defining the XGB hyper-parameter grid
    XGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 700, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.951, step = 0.05),
                     'min_split_loss': trial.suggest_int('min_split_loss', 0, 5, 1),
                     'max_depth' : trial.suggest_int('max_depth', 3, 7, 1),
                     'min_child_weight' : trial.suggest_int('min_child_weight', 5, 9, 1),
                     'subsample' : trial.suggest_float('subsample', 0.6, 1, step = 0.1),
                     'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.1)}
    
    ## Building the XGBRegressor model
    model = XGBRegressor(**XGB_param_grid, n_jobs = -1).fit(X_train, Y_train)
        
    ## Predicting on the test data-frame
    XGB_pred_test = model.predict(X_test)
    
    ## Evaluating model performance on the test set
    abs_diff = -np.mean(abs(XGB_pred_test - Y_test))
    
    ## Returning absolute difference of model test predictions
    return abs_diff

## Calling Optuna objective function
study = optuna.create_study(direction = 'maximize')
study.optimize(objective_amex, n_trials = 100)
