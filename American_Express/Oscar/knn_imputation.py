import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

## Reading data-file
delinquency_data = pd.read_csv('Delinquency_Features.csv')

## Defining buckets of variables
buckets = ['D_39', 'D_41', 'D_44', 'D_47', 'D_51', 'D_52', 'D_54', 'D_58', 'D_59', 'D_60',
           'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_74', 'D_75', 'D_78', 'D_79', 'D_80',
           'D_81', 'D_83', 'D_84', 'D_86', 'D_89', 'D_91', 'D_92', 'D_93', 'D_94', 'D_96',
           'D_102', 'D_103', 'D_104', 'D_107', 'D_109', 'D_112', 'D_113', 'D_115', 'D_118',
           'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127', 'D_128', 'D_129',
           'D_130', 'D_131', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144', 'D_145']

## Removing features with inf
data = delinquency_data.drop(columns = ['customer_ID', 'target'], axis = 1)
to_remove = data.columns.to_series()[np.isinf(data).any()]
delinquency_data = delinquency_data.drop(columns = [to_remove.index], axis = 1)

## Extracting features names
features = list(delinquency_data.columns)

## Looping to identify features with nan and backfill them with KNNImputer
for i in range(0, len(buckets)):
    
    ## Subsetting the bucket of features
    to_select = [x for x in features if x.startswith(buckets[i])]
    data_temp = delinquency_data[to_select] 
    
    
    