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

dtype_dict = {'customer_ID': 'object', 'R_1': 'float16', 'R_10': 'float16'}

## Reading data-files
test = pd.read_csv(file_content_stream, usecols = ['customer_ID', 'R_1', 'R_10'], dtype = dtype_dict)

## Computing basic summary-stats features
def summary_stats(x):
    
    d = {}
    d['R_1_mean'] = x['R_1'].mean()
    d['R_1_std'] = np.where(x['R_1'].shape[0] == 1, 0, np.std(x['R_1'], ddof = 1))
    d['R_1_max'] = x['R_1'].max()
    d['R_1_range'] = np.where(x['R_1'].shape[0] == 1, 0, x['R_1'].max() - x['R_1'].min())
    d['R_1_last_value'] = x['R_1'].iloc[-1]
    d['R_1_IQR'] = np.where(x['R_1'].shape[0] == 1, 0, np.percentile(x['R_1'], 75) - np.percentile(x['R_1'], 25))
    d['R_10_mean'] = x['R_10'].mean()
    d['R_10_max'] = x['R_10'].max()
    d['R_10_std'] = np.where(x['R_10'].shape[0] == 1, 0, np.std(x['R_10'], ddof = 1))
    d['R_10_range'] = np.where(x['R_10'].shape[0] == 1, 0, x['R_10'].max() - x['R_10'].min())
    
#     d['R_44_mean'] = x['R_44'].mean()
#     d['R_44_max'] = x['R_44'].max()
#     d['R_75_max'] = x['R_75'].max()
#     d['R_75_mean'] = x['R_75'].mean()
#     d['R_78_max'] = x['R_78'].max()
#     d['R_78_mean'] = x['R_78'].mean()
#     d['R_78_range'] = np.where(x['R_78'].shape[0] == 1, 0, x['R_78'].max() - x['R_78'].min())
#     d['R_44_std'] = np.where(x['R_44'].shape[0] == 1, 0, np.std(x['R_44'], ddof = 1))
#     d['R_75_median'] = x['R_75'].median()
#     d['R_78_std'] = np.where(x['R_78'].shape[0] == 1, 0, np.std(x['R_78'], ddof = 1))
#     d['R_74_mean'] = x['R_74'].mean()
#     d['R_44_range'] = np.where(x['R_44'].shape[0] == 1, 0, x['R_44'].max() - x['R_44'].min())
#     d['R_44_min'] = x['R_44'].min()
#     d['R_84_mean'] = x['R_84'].mean()
#     d['R_74_max'] = x['R_74'].max()
#     d['R_41_range'] = np.where(x['R_41'].shape[0] == 1, 0, x['R_41'].max() - x['R_41'].min())
#     d['R_75_min'] = x['R_75'].min()
#     d['R_44_IQR'] = np.where(x['R_44'].shape[0] == 1, 0, np.percentile(x['R_44'], 75) - np.percentile(x['R_44'], 25))
#     d['R_84_range'] = np.where(x['R_84'].shape[0] == 1, 0, x['R_84'].max() - x['R_84'].min())
    
    return pd.Series(d, index = ['R_1_mean', 'R_1_std', 'R_1_max', 'R_1_range', 
                                 'R_1_last_value', 'R_1_IQR', 'R_10_mean', 'R_10_max',
                                 'R_10_std', 'R_10_range'])

test_features = test.groupby('customer_ID').apply(summary_stats)
test_features['customer_ID'] = test_features.index
test_features = test_features.reset_index(drop = True)

## Uploading results in s3
test_features.to_csv('test_risk_features.csv', index = False)

sess.uploaR_data(path = 'test_risk_features.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')

