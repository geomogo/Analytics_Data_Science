import boto3
import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

import os
import sagemaker

sess = sagemaker.Session()

## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'AmericanExpress/Delinquency_Features_Imputed.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading data-file
delinquency_data = pd.read_csv(file_content_stream)
customer_ID_target = delinquency_data[['customer_ID', 'target']]
