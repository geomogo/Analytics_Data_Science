import boto3
import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'AmericanExpress/Risk_Features.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

## Reading data-files
risk_data = pd.read_csv(file_content_stream_1)
customer_ID = risk_data['customer_ID']

## Removing features with inf
data = risk_data.drop(columns = ['customer_ID'], axis = 1)
to_remove = data.columns.to_series()[np.isinf(data).any()]
risk_data = data.drop(columns = to_remove.index, axis = 1)

## Removing R_27 features
current_features = risk_data.columns
D_27_features = [x for x in current_features if x.startswith('R_27')]
risk_data = risk_data.drop(columns = D_27_features, axis = 1)

## Defining buckets of variables
buckets = ['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_10', 'R_11', 'R_12',
           'R_13', 'R_14', 'R_15', 'R_16', 'R_17', 'R_18', 'R_19', 'R_20', 'R_21', 'R_22',
           'R_23', 'R_24', 'R_25', 'R_28']

## Extracting features names
features = list(risk_data.columns)

## Looping to identify features with nan and backfill them with KNNImputer
for i in range(0, len(buckets)):
    print(buckets[i])
    ## Subsetting the bucket of features
    to_select = [x for x in features if x.startswith(buckets[i])]
    data_temp = risk_data[to_select] 
    
    ## Checkinig for nan
    to_check = data_temp.isna().any().sum()
    
    if (to_check > 0):
        
        imputed_data = KNNImputer(n_neighbors = 5).fit_transform(data_temp)
        n = imputed_data.shape[1]
        
        for i in range(0, n): 
            
            risk_data.loc[:, to_select[i]] = imputed_data[:, i]
                
    else:
        
        continue 
        
## Storing results in s3
risk_data = pd.concat([customer_ID, risk_data], axis = 1)
risk_data.to_csv('Risk_Features_Imputed.csv', index = False)

sess.upload_data(path = 'Risk_Features_Imputed.csv', 
                 bucket = bucket_name,
                 key_prefix = 'AmericanExpress')
