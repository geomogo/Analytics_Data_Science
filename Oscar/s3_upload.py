import boto3
import pandas as pd
import numpy as np

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Reading data-file
# risk_features = pd.read_csv('Risk_Features.csv')

## Uploading results in s3
sess.upload_data(path = 'Risk_Features.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')