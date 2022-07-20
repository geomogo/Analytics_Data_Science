import boto3
import pandas as pd; pd.set_option('display.max_columns', 200)
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


# Reading previous balance features
balance_features = pd.read_csv('Balance_Features.csv')
target = pd.read_csv("target.csv")


B4 = balance_features[['customer_ID', 'B_4', 'target']]

def summary_stats(x):
    
    b = {}
    b['B_4_mean'] = x['B_4'].mean()
    b['B_4_median'] = x['B_4'].median()
    b['B_4_min'] = x['B_4'].min()
    b['B_4_max'] = x['B_4'].max()
    b['B_4_range'] = x['B_4'].max() - x['B_4'].min()
    b['B_4_IQR'] = np.percentile(x['B_4'], 75) - np.percentile(x['B_4'], 25)
#     b['B_1_negative_count'] = np.sum(x['B_4'] < 0) 
    b['B_4_positive_count'] = np.sum(x['B_4'] > 0)
    b['B_4_values_above_mean'] = np.sum(x['B_4'] > x['B_4'].mean())
    
    return pd.Series(b, index = ['B_4_mean', 'B_4_median', 'B_4_min', 'B_4_max', 'B_4_range', 'B_4_IQR', 'B_4_negative_count', 'B_4_positive_count', 'B_4_values_above_mean'])


# Applying function to dataset
data_out = B4.groupby('customer_ID').apply(summary_stats)
data_out['customer_ID'] = data_out.index
data_out = data_out.reset_index(drop = True)

## Joining the to datasets
data_out = pd.merge(target, data_out, on = 'customer_ID', how = 'left')
data_out = data_out.drop(columns = ['target'], axis = 1)

# Aggregating new features to the dataset 
balance_features = pd.merge(balance_features, data_out, on = 'customer_ID', how = 'left')


# Exporting as csv
balance_features.to_csv('Balance_Features.csv', index = False)

print("finished")