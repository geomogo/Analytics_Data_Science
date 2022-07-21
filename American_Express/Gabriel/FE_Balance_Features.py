import boto3
import pandas as pd; pd.set_option('display.max_columns', 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'gabriel-data-science'
bucket = s3.Bucket(bucket_name)


## Defining files names
file_key_1 = 'AmericanExpress/Balance_Features.csv'
bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

balance_features = pd.read_csv(file_content_stream_1)

# Reading previous balance features
#balance_features = pd.read_csv('Balance_Features.csv')
target = pd.read_csv("target.csv")


B_13 = balance_features[['customer_ID', 'B_13', 'target']]

def summary_stats(x):
    
    b = {}
    b['B_13_mean'] = x['B_13'].mean()
    b['B_13_median'] = x['B_13'].median()
    b['B_13_min'] = x['B_13'].min()
    b['B_13_max'] = x['B_13'].max()
    b['B_13_range'] = x['B_13'].max() - x['B_13'].min()
    b['B_13_IQR'] = np.percentile(x['B_13'], 75) - np.percentile(x['B_13'], 25)
#     b['B_1_negative_count'] = np.sum(x['B_13'] < 0) 
    b['B_13_positive_count'] = np.sum(x['B_13'] > 0)
    b['B_13_values_above_mean'] = np.sum(x['B_13'] > x['B_13'].mean())
    
    return pd.Series(b, index = ['B_13_mean', 'B_13_median', 'B_13_min', 'B_13_max', 'B_13_range', 'B_13_IQR', 'B_13_negative_count', 'B_13_positive_count', 'B_13_values_above_mean'])


# Applying function to dataset
data_out = B_13.groupby('customer_ID').apply(summary_stats)
data_out['customer_ID'] = data_out.index
data_out = data_out.reset_index(drop = True)

## Joining the to datasets
data_out = pd.merge(target, data_out, on = 'customer_ID', how = 'left')
data_out = data_out.drop(columns = ['target'], axis = 1)

# Aggregating new features to the dataset 
balance_features = pd.merge(balance_features, data_out, on = 'customer_ID', how = 'left')

#balance_features = balance_features.drop(columns = ['B_29', 'B_39', 'B_42'])


# Exporting as csv
balance_features.to_csv('Balance_Features.csv', index = False)

sess.upload_data(path = 'Balance_Features.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')




#sess.upload_data(path = 'Balance_Features.csv', 
#                 bucket = bucket_name,
#                 key_prefix = 'AmericanExpress')

print("Finished")