import boto3
import pandas as pd; pd.set_option('display.max_columns', 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sagemaker

sess = sagemaker.Session()

# Reading previous balance features
balance_features = pd.read_csv('Balance_Features.csv')
target = pd.read_csv("target.csv")


B5 = balance_features[['customer_ID', 'B_5', 'target']]

def summary_stats(x):
    
    b = {}
    b['B_5_mean'] = x['B_5'].mean()
    b['B_5_median'] = x['B_5'].median()
    b['B_5_min'] = x['B_5'].min()
    b['B_5_max'] = x['B_5'].max()
    b['B_5_range'] = x['B_5'].max() - x['B_5'].min()
    b['B_5_IQR'] = np.percentile(x['B_5'], 75) - np.percentile(x['B_5'], 25)
#     b['B_1_negative_count'] = np.sum(x['B_5'] < 0) 
    b['B_5_positive_count'] = np.sum(x['B_5'] > 0)
    b['B_5_values_above_mean'] = np.sum(x['B_5'] > x['B_5'].mean())
    
    return pd.Series(b, index = ['B_5_mean', 'B_5_median', 'B_5_min', 'B_5_max', 'B_5_range', 'B_5_IQR', 'B_5_negative_count', 'B_5_positive_count', 'B_5_values_above_mean'])


# Applying function to dataset
data_out = B5.groupby('customer_ID').apply(summary_stats)
data_out['customer_ID'] = data_out.index
data_out = data_out.reset_index(drop = True)

## Joining the to datasets
data_out = pd.merge(target, data_out, on = 'customer_ID', how = 'left')
data_out = data_out.drop(columns = ['target'], axis = 1)

# Aggregating new features to the dataset 
balance_features = pd.merge(balance_features, data_out, on = 'customer_ID', how = 'left')


# Exporting as csv
balance_features.to_csv('Balance_Features.csv', index = False)

#sess.upload_data(path = 'Balance_Features.csv', 
#                 bucket = bucket_name,
#                 key_prefix = 'AmericanExpress')