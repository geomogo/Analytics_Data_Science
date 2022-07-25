import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
# from xgboost import XGBClassifier
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
# file_key_3 = 'AmericanExpress/amex_train_payment_spend_final.csv'
# file_key_4 = 'AmericanExpress/amex_test_payment_spend_final.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

# bucket_object_3 = bucket.Object(file_key_3)
# file_object_3 = bucket_object_3.get()
# file_content_stream_3 = file_object_3.get('Body')

# bucket_object_4 = bucket.Object(file_key_4)
# file_object_4 = bucket_object_4.get()
# file_content_stream_4 = file_object_4.get('Body')

## Reading data-files
oscar_data_train = pd.read_csv(file_content_stream_1, 
                               usecols = ['D_44_median', 'D_44_mean', 'D_44_max', 
                                          'D_75_max', 'D_75_mean', 'D_78_max', 
                                          'D_78_mean', 'D_78_range', 'D_44_std', 
                                          'D_75_median', 'D_78_std', 'D_74_mean',
                                          'D_44_range', 'D_44_min', 'D_84_mean', 
                                          'D_74_max', 'D_41_range', 'D_75_min', 
                                          'D_44_IQR', 'D_84_range', 'target'])

# oscar_data_train = pd.read_csv(file_content_stream_1, 
#                                usecols = ['D_44_median', 'D_44_mean', 'D_44_max', 
#                                           'D_75_max', 'D_75_mean', 'D_78_max', 
#                                           'D_78_mean', 'D_78_range', 'D_44_std', 
#                                           'D_75_median', 'D_78_std', 'D_74_mean',
#                                           'D_44_range', 'D_44_min', 'D_84_mean', 
#                                           'D_74_max', 'D_41_range', 'D_75_min', 
#                                           'D_44_IQR', 'D_84_range', 'target',
#                                           'customer_ID'])

oscar_data_test = pd.read_csv(file_content_stream_2)

# evan_data_train = pd.read_csv(file_content_stream_3)
# evan_data_train = evan_data_train.drop(columns = ['target'], axis = 1)

# evan_data_test = pd.read_csv(file_content_stream_4)

## Joining datasets
# data_train = pd.merge(oscar_data_train, evan_data_train, on = 'customer_ID', how = 'left')
# data_test = pd.merge(oscar_data_test, evan_data_test, on = 'customer_ID', how = 'left')

## Defining input and target 
# X = data_train.drop(columns = 'target', axis = 1)
# Y = data_train['target']
X = oscar_data_train.drop(columns = 'target', axis = 1)
Y = oscar_data_train['target']

## Spliting the data into train, validation, and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y)
dtrain = lgb.Dataset(X_train, label = Y_train)

############
## Optuna ##
############

def objective_amex(trial):
    
    ## Defining the XGB hyper-parameter grid
    LGB_param_grid = {'objective': 'binary',
                      'metric': 'binary_logloss',
                      'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                      'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                      'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                      'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                      'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                      'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                      'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
                     }
    
    ## Building the LightGBM model
    model = lgb.train(param, dtrain)
        
    ## Predicting on the test data-frame
    LGB_pred_test = model.predict_proba(X_test)[:, 1]
    
    ## Evaluating model performance on the test set
    amex_score = amex_metric(Y_test, LGB_pred_test)
    
    ## Returning absolute difference of model test predictions
    return amex_score

## Calling Optuna objective function
study = optuna.create_study(direction = 'maximize')
study.optimize(objective_amex, n_trials = 100)

## Extracting best model 
best_params = study.best_trial.params
LGB_md = lgb(best_param, dtrain)

## Predicting on test 
# X_test_real = data_test.drop(columns = ['customer_ID'], axis = 1)
# X_test_real_pred = LGB_md.predict_proba(X_test_real)[:, 1]

# data_out = pd.DataFrame({'customer_ID': data_test['customer_ID'], 'prediction': X_test_real_pred})

X_test_real = oscar_data_test.drop(columns = ['customer_ID'], axis = 1)
X_test_real_pred = LGB_md.predict_proba(X_test_real)[:, 1]

data_out = pd.DataFrame({'customer_ID': oscar_data_test['customer_ID'], 'prediction': X_test_real_pred})


## Uploading results in s3
data_out.to_csv('submission.csv', index = False)

sess.upload_data(path = 'submission.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')

