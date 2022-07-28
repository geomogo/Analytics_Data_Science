import boto3
import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

import os
import sagemaker

sess = sagemaker.Session()

## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'gabriel-data-science'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'AmericanExpress/Balance_Features.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

## Reading data-file
Balance_Features = pd.read_csv(file_content_stream_1)
customer_ID_target = Balance_Features[['customer_ID', 'target']]

## Defining buckets of variables
buckets = ['B_13_IQR', 'B_8_IQR', 'B_13', 'B_8', 'B_15_IQR', 'B_19_IQR', 'B_22_IQR', 'B_26_IQR', 'B_20_IQR', 'B_16_IQR', 'B_27_IQR', 'B_33_IQR', 'B_2_IQR', 'B_3_IQR', 'B_25', 'B_15', 'B_8_min', 'B_8_mean', 'B_8_max', 'B_8_median', 'B_8_range', 'B_15_max', 'B_15_min', 'B_15_range', 'B_15_median', 'B_15_mean', 'B_13_mean', 'B_13_range', 'B_13_max', 'B_13_median', 'B_13_min', 'B_22', 'B_2', 'B_19', 'B_20', 'B_26', 'B_3', 'B_16', 'B_33', 'B_27', 'B_41', 'B_40_IQR', 'B_6_IQR', 'B_6', 'B_40', 'B_37', 'B_22_max', 'B_22_range', 'B_26_max', 'B_19_mean', 'B_26_min', 'B_26_median', 'B_26_mean', 'B_22_min', 'B_16_min', 'B_20_range', 'B_22_median', 'B_16_median', 'B_16_max', 'B_16_mean', 'B_19_max', 'B_19_range', 'B_27_max', 'B_22_mean', 'B_27_min', 'B_20_mean', 'B_27_median', 'B_20_min', 'B_20_max', 'B_27_mean', 'B_19_min', 'B_19_median', 'B_20_median', 'B_6_max', 'B_6_median', 'B_6_range', 'B_6_min', 'B_3_mean', 'B_6_mean', 'B_2_range', 'B_33_max', 'B_33_min', 'B_33_median', 'B_33_mean', 'B_2_mean', 'B_2_min', 'B_2_max', 'B_2_median', 'B_3_median', 'B_3_min', 'B_3_max', 'B_3_range', 'B_11_IQR', 'B_17_positive_count', 'B_19_values_above_mean', 'B_19_positive_count', 'B_11_mean', 'B_11_median', 'B_11_min', 'B_11_positive_count', 'B_11_values_above_mean', 'B_21_median', 'B_11_max', 'B_11_range', 'B_17_values_above_mean', 'B_21_mean', 'customer_ID', 'B_21_min', 'B_24_positive_count', 'B_18_positive_count', 'B_18_IQR', 'B_18_max', 'B_18_min', 'B_18_median', 'B_18_mean', 'B_16_values_above_mean', 'B_16_positive_count', 'B_27_values_above_mean', 'B_27_positive_count', 'B_26_values_above_mean', 'B_26_positive_count', 'B_24_values_above_mean', 'B_24_IQR', 'B_21_max', 'B_24_range', 'B_15_values_above_mean', 'B_24_min', 'B_24_median', 'B_24_mean', 'B_22_values_above_mean', 'B_22_positive_count', 'B_20_values_above_mean', 'B_20_positive_count', 'B_21_values_above_mean', 'B_21_positive_count', 'B_21_IQR', 'B_21_range', 'B_24_max', 'B_10_median', 'B_15_positive_count', 'B_10_max', 'B_10_values_above_mean', 'B_10_min', 'B_18_values_above_mean', 'B_10_range', 'B_10_positive_count', 'B_10_IQR']


## Removing features with inf
data = Balance_Features.drop(columns = ['customer_ID', 'target'], axis = 1)
to_remove = data.columns.to_series()[np.isinf(data).any()]
Balance_Features = data.drop(columns = to_remove.index, axis = 1)

## Extracting features names
features = list(Balance_Features.columns)

## Looping to identify features with nan and backfill them with KNNImputer
for i in range(0, len(buckets)):
    print(buckets[i])
    ## Subsetting the bucket of features
    to_select = [x for x in features if x.startswith(buckets[i])]
    data_temp = Balance_Features[to_select] 
    
    ## Checkinig for nan
    to_check = data_temp.isna().any().sum()
    
    if (to_check > 0):
        
        imputed_data = KNNImputer(n_neighbors = 5).fit_transform(data_temp)
        n = imputed_data.shape[1]
        
        for i in range(0, n): 
            
            Balance_Features.loc[:, to_select[i]] = imputed_data[:, i]
                
    else:
        
        continue 
        
## Storing results in s3
Balance_Features = pd.concat([customer_ID_target, Balance_Features], axis = 1)
Balance_Features.to_csv('Balance_Features_Imputed.csv', index = False)

sess.upload_data(path = 'Balance_Features_Imputed.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')

