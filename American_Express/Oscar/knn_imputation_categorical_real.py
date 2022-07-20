import boto3
import pandas as pd 
import numpy as np

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'AmericanExpress/Delinquency_Features_Imputed.csv'
file_key_2 = 'AmericanExpress/Delinquency_Features_Categorical_Last.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Reading data-files
data_left = pd.read_csv(file_content_stream_1)
data_right = pd.read_csv(file_content_stream_2)

## Merging datasets
data = pd.merge(data_left, data_right, on = 'customer_ID', how = 'left')

## Defining target features
target = ['D_66_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last']

## Defining input variables 
X = data.drop(columns = ['customer_ID', 'target', 'D_63_last', 'D_64_last', 'D_66_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last', 'D_120_last'], axis = 1)

## Looping to backfill missing values
for i in range(0, len(target)):
    
    