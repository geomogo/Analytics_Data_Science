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
file_key = 'AmericanExpress/test_data.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

dtype_dict = {'customer_ID': 'object', 'D_39': 'float16', 'D_41': 'float16',
              'D_44': 'float16', 'D_52': 'float16', 'D_58': 'float16', 
              'D_65': 'float16', 'D_74': 'float16', 'D_75': 'float16',
              'D_78': 'float16', 'D_84': 'float16'}

## Reading data-files
test = pd.read_csv(file_content_stream, use_cols = ['customer_ID', 'D_39', 'D_41', 'D_44', 'D_52', 'D_58', 'D_65', 'D_74','D_75', 'D_78', 'D_84'], dtype = dtype_dict)

## Computing basic summary-stats features
def summary_stats(x):
    
    d = {}
    d['D_44_median'] = x['D_44'].median()
    d['D_44_mean'] = x['D_44'].mean()
    d['D_44_max'] = x['D_44'].max()
    d['D_75_max'] = x['D_75'].max()
    d['D_75_mean'] = x['D_75'].mean()
    d['D_78_max'] = x['D_78'].max()
    d['D_78_mean'] = x['D_78'].mean()
    d['D_78_range'] = np.where(x['D_78'].shape[0] == 1, 0, x['D_78'].max() - x['D_78'].min())
    d['D_44_std'] = np.where(x['D_44'].shape[0] == 1, 0, np.std(x['D_44'], ddof = 1))
    d['D_75_median'] = x['D_75'].median()
    d['D_78_std'] = np.where(x['D_78'].shape[0] == 1, 0, np.std(x['D_78'], ddof = 1))
    d['D_74_mean'] = x['D_74'].mean()
    d['D_44_range'] = np.where(x['D_44'].shape[0] == 1, 0, x['D_44'].max() - x['D_44'].min())
    d['D_44_min'] = x['D_44'].min()
    d['D_84_mean'] = x['D_84'].mean()
    d['D_74_max'] = x['D_74'].max()
    d['D_41_range'] = np.where(x['D_41'].shape[0] == 1, 0, x['D_41'].max() - x['D_41'].min())
    d['D_75_min'] = x['D_75'].min()
    d['D_44_IQR'] = np.where(x['D_44'].shape[0] == 1, 0, np.percentile(x['D_44'], 75) - np.percentile(x['D_44'], 25))
    d['D_84_range'] = np.where(x['D_84'].shape[0] == 1, 0, x['D_84'].max() - x['D_84'].min())
    
    return pd.Series(d, index = ['D_44_median', 'D_44_mean', 'D_44_max', 'D_75_max', 
                                 'D_75_mean', 'D_78_max', 'D_78_mean', 'D_78_range',
                                 'D_44_std', 'D_75_median', 'D_78_std', 'D_74_mean',
                                 'D_44_range', 'D_44_min', 'D_84_mean', 'D_74_max',
                                 'D_41_range', 'D_75_min', 'D_44_IQR', 'D_84_range'])

test_features = test.groupby('customer_ID').apply(summary_stats)
test_features['customer_ID'] = test_features.index
test_features = test_features.reset_index(drop = True)

## Uploading results in s3
test_features.to_csv('test_delinquency_features.csv', index = False)

sess.upload_data(path = 'test_delinquency_features.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')

