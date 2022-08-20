import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'Store-Sales-Time-Series-Forecast/oil.csv'
file_key_2 = 'Store-Sales-Time-Series-Forecast/holidays_events.csv'
file_key_3 = 'Store-Sales-Time-Series-Forecast/stores.csv'
file_key_4 = 'Store-Sales-Time-Series-Forecast/transactions.csv'
file_key_5 = 'Store-Sales-Time-Series-Forecast/train.csv'
file_key_6 = 'Store-Sales-Time-Series-Forecast/test.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

bucket_object_3 = bucket.Object(file_key_3)
file_object_3 = bucket_object_3.get()
file_content_stream_3 = file_object_3.get('Body')

bucket_object_4 = bucket.Object(file_key_4)
file_object_4 = bucket_object_4.get()
file_content_stream_4 = file_object_4.get('Body')

bucket_object_5 = bucket.Object(file_key_5)
file_object_5 = bucket_object_5.get()
file_content_stream_5 = file_object_5.get('Body')

bucket_object_6 = bucket.Object(file_key_6)
file_object_6 = bucket_object_6.get()
file_content_stream_6 = file_object_6.get('Body')

## Reading data-files
oil = pd.read_csv(file_content_stream_1)
holidays = pd.read_csv(file_content_stream_2)
stores = pd.read_csv(file_content_stream_3)
transactions = pd.read_csv(file_content_stream_4)
train = pd.read_csv(file_content_stream_5)
test = pd.read_csv(file_content_stream_6)

## Changing feature labels in holidays and stores
holidays.columns = ['date', 'holiday_type', 'locale', 'locale_name', 'description', 'transferred']
stores.columns = ['store_nbr', 'city', 'state', 'store_type', 'cluster']

## Consolidating train data
train = pd.merge(train, oil, on = 'date', how = 'left')
train = pd.merge(train, holidays, on = 'date', how = 'left')
train = pd.merge(train, stores, on = 'store_nbr', how = 'left')
train['date'] = pd.to_datetime(train['date'], format = '%Y-%m-%d')

## Consolidating test data
test = pd.merge(test, holidays, on = 'date', how = 'left')
test = pd.merge(test, oil, on = 'date', how = 'left')
test = pd.merge(test, stores, on = 'store_nbr', how = 'left')
test['date'] = pd.to_datetime(test['date'], format = '%Y-%m-%d')

## Basic feature engineering 
train['day'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
train['is_holiday'] = np.where(train['holiday_type'] == 'Holiday', 1, 0)
train = train[['onpromotion', 'dcoilwtico', 'is_holiday', 'day', 'month', 'family', 'store_type', 'cluster', 'sales']]
train_dummies_1 = pd.get_dummies(train['family'])
train_dummies_2 = pd.get_dummies(train['store_type'])
train_dummies_3 = pd.get_dummies(train['cluster'])
train = pd.concat([train.drop(columns = ['family', 'store_type', 'cluster'], axis = 1), train_dummies_1, train_dummies_2, train_dummies_3], axis = 1)

test['day'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
test['is_holiday'] = np.where(test['holiday_type'] == 'Holiday', 1, 0)
test_ids = test['id']
test = test[['onpromotion', 'dcoilwtico', 'is_holiday', 'day', 'month', 'family', 'store_type', 'cluster']]
test_dummies_1 = pd.get_dummies(test['family'])
test_dummies_2 = pd.get_dummies(test['store_type'])
test_dummies_3 = pd.get_dummies(test['cluster'])
test = pd.concat([test.drop(columns = ['family', 'store_type', 'cluster'], axis = 1), test_dummies_1, test_dummies_2, test_dummies_3], axis = 1)

X = train.drop(columns = ['sales'], axis = 1)
Y = train['sales']

## Defining the hyper-parameter grid
LightGBM_param_grid = {'n_estimators': [100, 300, 500],
                       'learning_rate': [0.01, 0.1, 1],
                       'max_depth': [5, 7, 10],
                       'num_leaves': [2, 5, 10, 15],
                       'min_data_in_leaf': [10, 20, 50],
                       'min_gain_to_split': [1, 5, 10],
                       'lambda_l1': [0, 1, 10, 100],
                       'lambda_l2': [0, 1, 10, 100]
                       }

## Performing grid search with 5 folds
LightGBM_grid_search = GridSearchCV(LGBMRegressor(), LightGBM_param_grid, cv = 5, scoring = 'neg_mean_squared_log_error', n_jobs = -1, verbose = 4).fit(X, Y)

## Extracting the best score
best_score = LightGBM_grid_search.best_score_
print('The best mean squared log error:', best_score)

## Extracting the best model
LightGBM_md = LightGBM_grid_search.best_estimator_

## Predicting on test with best xgboost model 
LightGBM_pred = LightGBM_md.predict(test)
    
## Defining data-frame to store results
data_out = pd.DataFrame({'id': test_ids, 'sales': LightGBM_pred})
data_out.to_csv('LightGBM_baseline_run.csv', index = False)
