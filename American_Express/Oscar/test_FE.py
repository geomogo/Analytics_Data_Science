import boto3
import pandas as pd
import numpy as np

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'AmericanExpress/test_data.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading data-files
data = pd.read_csv(file_content_stream, use_cols = ['D_39', 'D_41', 'D_44', 'D_52', 'D_58', 'D_65', 'D_74','D_75', 'D_78', 'D_84'])

