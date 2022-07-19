import boto3
import pandas as pd 
import numpy as np
import os
from sklearn.impute import KNNImputer

## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

# ## Defining files names
# file_key = 'AmericanExpress/Delinquency_Features.csv'

# # bucket_object = bucket.Object(file_key)
# file_object = bucket_object.get()
# file_content_stream = file_object.get('Body')

# ## Reading data-file
# delinquency_data = pd.read_csv(file_content_stream)
# customer_ID_target = delinquency_data[['customer_ID', 'target']]

# ## Defining buckets of variables
# buckets = ['D_39', 'D_41', 'D_44', 'D_47', 'D_51', 'D_52', 'D_54', 'D_58', 'D_59', 'D_60',
#            'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_74', 'D_75', 'D_78', 'D_79', 'D_80',
#            'D_81', 'D_83', 'D_84', 'D_86', 'D_89', 'D_91', 'D_92', 'D_93', 'D_94', 'D_96',
#            'D_102', 'D_103', 'D_104', 'D_107', 'D_109', 'D_112', 'D_113', 'D_115', 'D_118',
#            'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127', 'D_128', 'D_129',
#            'D_130', 'D_131', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144', 'D_145']

# ## Removing features with inf
# data = delinquency_data.drop(columns = ['customer_ID', 'target'], axis = 1)
# to_remove = data.columns.to_series()[np.isinf(data).any()]
# delinquency_data = delinquency_data.drop(columns = to_remove.index, axis = 1)

# ## Extracting features names
# features = list(delinquency_data.columns)

# ## Looping to identify features with nan and backfill them with KNNImputer
# for i in range(0, len(buckets)):
#     print(buckets[i])
#     ## Subsetting the bucket of features
#     to_select = [x for x in features if x.startswith(buckets[i])]
#     data_temp = delinquency_data[to_select] 
    
#     ## Checkinig for nan
#     to_check = data_temp.isna().any().sum()
    
#     if (to_check > 0):
        
#         imputed_data = KNNImputer(n_neighbors = 5).fit_transform(data_temp)
#         n = imputed_data.shape[1]
        
#         for i in range(0, n): 
            
#             delinquency_data.loc[:, to_select[i]] = imputed_data[:, i]
                
#     else:
        
#         continue 
        
# ## Storing results in s3
# delinquency_data = pd.concat([customer_ID_target, delinquency_data], axis = 1)
# delinquency_data.to_csv('Delinquency_Features_Imputed.csv', index = False)
# s3.meta.client.upload_file('Delinquency_Features_Imputed.csv', 'YOUR_S3_BUCKET_NAME', 'DESIRED_S3_OBJECT_NAME')    

test = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
# test.to_csv('mytest.csv')

boto3.resource('s3').Bucket(bucket).Object(os.path.join(bucket_name, 'AmericanExpress', 'mytest.csv')).upload_fileobj(test)

# s3.meta.client.upload_file('mytest.csv', 'analytics-data-science-competitions/AmericanExpress', 'mytest.csv')    
