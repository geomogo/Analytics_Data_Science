import boto3
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'AmericanExpress/Balance_Features.csv'
file_key_2 = 'AmericanExpress/Delinquency_Features_Categorical_Last.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Reading data-files
data_left = pd.read_csv(file_content_stream_1)
to_remove = data_left.drop(columns = ['customer_ID', 'target'], axis = 1).columns.to_series()[np.isnan(data_left.drop(columns = ['customer_ID', 'target'], axis = 1)).any()]
data_left = data_left.drop(columns = to_remove.index, axis = 1)

data_right = pd.read_csv(file_content_stream_2)

## Merging datasets
data = pd.merge(data_left, data_right, on = 'customer_ID', how = 'left')
data = data.drop(columns = ['D_66_last'], axis = 1)

## Defining target features
target = ["B_30", "B_38"]

## Defining input variables 
X = data.drop(columns = ['customer_ID', 'target', 'D_63_last', 'D_64_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last', 'D_126_last'], axis = 1)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))

## Looping to backfill missing values
for i in range(0, len(target)):
    print(target[i])
    ## Defining the target variable
    Y = data[target[i]]
    Y_full = Y[~np.isnan(Y)]
    X_full = X[~np.isnan(Y)]
    
    X_to_be_filled = X[np.isnan(Y)]
    
    ## Defining the model
    knn_md = KNeighborsClassifier(n_neighbors = 5).fit(X_full, Y_full)
    
    ## Predicting 
    data.loc[np.isnan(Y), target[i]] = knn_md.predict(X_to_be_filled)
    
## Uploading results in s3
data.to_csv('Delinquency_Features_Filled.csv', index = False)

sess.upload_data(path = 'Delinquency_Features_Filled.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')