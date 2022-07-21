import boto3
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from Amex_Metric import amex_metric

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'AmericanExpress/Delinquency_Features_Filled.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading data-files
data = pd.read_csv(file_content_stream)
data = data.drop(columns = ['D_64_last'], axis = 1)

## Putting variables in the right shape 
data['D_68_last'] = data['D_68_last'].astype(str)
data['D_114_last'] = data['D_114_last'].astype(str)
data['D_116_last'] = data['D_116_last'].astype(str)
data['D_117_last'] = data['D_117_last'].astype(str)
data['D_120_last'] = data['D_120_last'].astype(str)
data['D_126_last'] = data['D_126_last'].astype(str)

## Converting to dummies
dummies = pd.get_dummies(data[['D_63_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last', 'D_126_last']])

## Appeding dummies 
data = data.drop(columns = ['D_63_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last', 'D_126_last'], axis = 1)
data = pd.concat([data, dummies], axis = 1)

## Defining input and target variables
X = data.drop(columns = ['customer_ID', 'target'], axis = 1)
Y = data['target']

## Spliting the data into train, validation, and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.95, stratify = Y)

## Defining the customized scoring function 
amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

## Defining list to store results
features_to_select = list()

for i in tqdm(range(0, 10)):
    
    ## Running RFE with Random forest
    RF_auto_feature = RFECV(estimator = RandomForestClassifier(n_estimators = 50, max_depth = 3), step = 20, scoring = amex_function, min_features_to_select = 10, cv = 3, n_jobs = -1).fit(X_train, Y_train)

    ## Appending results 
    features_to_select.append(X_train.columns[RF_auto_feature.support_])
    
## Putting results as data-frame
features_to_select = pd.DataFrame(features_to_select)
features_to_select.to_csv('Delinquency_Features_to_select_1.csv', index = False)