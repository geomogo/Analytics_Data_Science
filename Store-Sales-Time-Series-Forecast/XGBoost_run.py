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
train = train[train['cluster'] == 3].reset_index(drop = True)

## Consolidating test data
test = pd.merge(test, holidays, on = 'date', how = 'left')
test = pd.merge(test, oil, on = 'date', how = 'left')
test = pd.merge(test, stores, on = 'store_nbr', how = 'left')
test['date'] = pd.to_datetime(test['date'], format = '%Y-%m-%d')
test = test[test['cluster'] == 3].reset_index(drop = True)

## Basic feature engineering 
train['day'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
train['is_holiday'] = np.where(train['holiday_type'] == 'Holiday', 1, 0)
train = train[['onpromotion', 'dcoilwtico', 'is_holiday', 'day', 'month', 'family', 'sales']]
train_dummies = pd.get_dummies(train['family'])
train = pd.concat([train.drop(columns = 'family', axis = 1), train_dummies], axis = 1)

test['day'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
test['is_holiday'] = np.where(test['holiday_type'] == 'Holiday', 1, 0)
test = test[['onpromotion', 'dcoilwtico', 'is_holiday', 'day', 'month', 'family']]
test_dummies = pd.get_dummies(test['family'])
test = pd.concat([test.drop(columns = 'family', axis = 1), test_dummies], axis = 1)



    
