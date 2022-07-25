import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import optuna
from xgboost import XGBClassifier
from Amex_Metric import amex_metric

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'AmericanExpress/Delinquency_Features_Filled.csv'
file_key_2 = 'AmericanExpress/test_delinquency_features.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Reading data-files
data = pd.read_csv(file_content_stream_1, usecols = ['D_44_median', 'D_44_mean', 'D_44_max', 
                                                     'D_75_max', 'D_75_mean', 'D_78_max', 
                                                     'D_78_mean', 'D_78_range', 'D_44_std', 
                                                     'D_75_median', 'D_78_std', 'D_74_mean',
                                                     'D_44_range', 'D_44_min', 'D_84_mean', 
                                                     'D_74_max', 'D_41_range', 'D_75_min', 
                                                     'D_44_IQR', 'D_84_range', 'target'])
test = pd.read_csv(file_content_stream_2)

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
    XGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 500, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.951, step = 0.05),
                     'min_split_loss': trial.suggest_int('min_split_loss', 0, 5, 1),
                     'max_depth' : trial.suggest_int('max_depth', 3, 7, 1),
                     'min_child_weight' : trial.suggest_int('min_child_weight', 5, 9, 1),
                     'subsample' : trial.suggest_float('subsample', 0.6, 1, step = 0.1),
                     'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.1)}
    
    ## Building the XGBRegressor model
    model = XGBClassifier(**XGB_param_grid, n_jobs = -1).fit(X_train, Y_train)
        
    ## Predicting on the test data-frame
#     XGB_pred_test = model.predict_proba(X_test)[:, 1]
    XGB_pred_test = model.predict(X_test)
    
    ## Evaluating model performance on the test set
#     amex_score = amex_metric(Y_test, XGB_pred_test)
    amex_score = accuracy_score(Y_test, XGB_pred_test)
    
    ## Returning absolute difference of model test predictions
    return amex_score

## Calling Optuna objective function
study = optuna.create_study(direction = 'maximize')
study.optimize(objective_amex, n_trials = 100)

## Extracting best model 
best_params = study.best_trial.params
XGB_md = XGBClassifier(**best_params, n_jobs = -1).fit(X_train, Y_train)

## Predicting on test 
X_test_real = test.drop(columns = ['customer_ID'], axis = 1)
X_test_real_pred = XGB_md.predict_proba(X_test_real)[:, 1]

data_out = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': X_test_real_pred})

## Uploading results in s3
data_out.to_csv('submission.csv', index = False)

sess.upload_data(path = 'submission.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')

