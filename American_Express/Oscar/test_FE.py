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
file_key_1 = 'AmericanExpress/test_data.csv'
# file_key_2 = 'AmericanExpress/test_delinquency_features.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

# bucket_object_2 = bucket.Object(file_key_2)
# file_object_2 = bucket_object_2.get()
# file_content_stream_2 = file_object_2.get('Body')

dtype_dict = {'customer_ID': 'object', 'D_39': 'float16', 'D_41': 'float16',
              'D_44': 'float16', 'D_52': 'float16', 'D_58': 'float16', 'D_59': 'float16',  
              'D_65': 'float16', 'D_74': 'float16', 'D_75': 'float16',
              'D_78': 'float16', 'D_84': 'float16'}

## Reading data-files
test = pd.read_csv(file_content_stream_1, 
                   usecols = ['customer_ID', 'D_39', 'D_41', 'D_44', 'D_52', 
                              'D_58', 'D_59', 'D_65', 'D_74','D_75', 'D_78', 
                              'D_84'], dtype = dtype_dict)

# delinquency_features = pd.read_csv(file_content_stream_2)

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
    
    d['D_41_std'] = np.where(x['D_41'].shape[0] == 1, 0, np.std(x['D_41'], ddof = 1))
    d['D_39_max'] = x['D_39'].max()
    d['D_84_max'] = x['D_84'].max()
    d['D_84_std'] = np.where(x['D_84'].shape[0] == 1, 0, np.std(x['D_84'], ddof = 1))
    d['D_74_median'] = x['D_74'].median()
    d['D_52_median'] = x['D_52'].median()
    d['D_58_median'] = x['D_58'].median()
    d['D_65_max'] = x['D_65'].max()
    d['D_58_max'] = x['D_58'].max()
    d['D_59_mean'] = x['D_59'].mean()
    
    return pd.Series(d, index = ['D_44_median', 'D_44_mean', 'D_44_max', 'D_75_max', 
                                 'D_75_mean', 'D_78_max', 'D_78_mean', 'D_78_range',
                                 'D_44_std', 'D_75_median', 'D_78_std', 'D_74_mean',
                                 'D_44_range', 'D_44_min', 'D_84_mean', 'D_74_max',
                                 'D_41_range', 'D_75_min', 'D_44_IQR', 'D_84_range',
                                 'D_41_std', 'D_39_max', 'D_84_max', 'D_84_std', 
                                 'D_74_median', 'D_52_median', 'D_58_median',
                                 'D_65_max', 'D_58_max', 'D_59_mean'])

test_features = test.groupby('customer_ID').apply(summary_stats)
test_features['customer_ID'] = test_features.index
test_features = test_features.reset_index(drop = True)

## Uploading results in s3
test_features.to_csv('test_delinquency_features.csv', index = False)

# delinquency_features = pd.merge(delinquency_features, test_features, on = 'customer_ID', how = 'left')
# delinquency_features.to_csv('test_delinquency_features.csv', index = False)

sess.upload_data(path = 'test_delinquency_features.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')

