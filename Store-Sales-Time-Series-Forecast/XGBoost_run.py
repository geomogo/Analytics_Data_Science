import boto3
import pandas as pd; pd.set_option('display.max_columns', 200)
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

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
train = train[['onpromotion', 'dcoilwtico', 'is_holiday', 'day', 'month', 'family', 'cluster', 'sales']]

test['day'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
test['is_holiday'] = np.where(test['holiday_type'] == 'Holiday', 1, 0)
test = test[['id', 'onpromotion', 'dcoilwtico', 'is_holiday', 'day', 'month', 'family', 'cluster']]

## Looping through the clusters
clusters = train['cluster'].unique()

## Defining list to store results
results = list()

for i in range(0, len(clusters)):
    
    print('Working on cluster ', i+1, 'out of', len(clusters))
    
    train_loop = train[train['cluster'] == clusters[i]]
    train_loop = train_loop.drop(columns = ['cluster'], axis = 1)
    train_dummies = pd.get_dummies(train_loop['family'])
    train_loop = pd.concat([train_loop.drop(columns = 'family', axis = 1), train_dummies], axis = 1)
    
    test_loop = test[test['cluster'] == clusters[i]]
    test_ids = test_loop['id']
    test_loop = test_loop.drop(columns = ['cluster', 'id'], axis = 1)
    test_dummies = pd.get_dummies(test_loop['family'])
    test_loop = pd.concat([test_loop.drop(columns = 'family', axis = 1), test_dummies], axis = 1)

    X = train_loop.drop(columns = ['sales'], axis = 1)
    Y = train_loop['sales']

    ## Defining the hyper-parameter grid
    XGBoost_param_grid = {'n_estimators': [300],
                          'max_depth': [5, 7],
                          'min_child_weight': [5, 7, 10],
                          'learning_rate': [0.01, 0.001],
                          'gamma': [0.3, 0.1],
                          'subsample': [0.8, 1],
                          'colsample_bytree': [0.8, 1]}

    ## Performing grid search with 5 folds
    XGBoost_grid_search = GridSearchCV(XGBRegressor(), XGBoost_param_grid, cv = 5, scoring = 'neg_mean_squared_log_error', n_jobs = -1, verbose = 4).fit(X, Y)

    ## Extracting the best score
    best_score = XGBoost_grid_search.best_score_
    print('The best mean squared log error:', best_score)

    ## Extracting the best model
    XGBoost_md = XGBoost_grid_search.best_estimator_

    ## Predicting on test with best xgboost model 
    xgb_pred = XGBoost_md.predict(test_loop)
    
    ## Defining data-frame to store results
    data_out = pd.DataFrame({'id': test_ids, 'sales': xgb_pred})
    
    ## Appending results
    results.append(data_out)

## Combining results as a single data-frame
results = pd.DataFrame(results)
results.to_csv('XGBoost_baseline_run.csv', index = False)
