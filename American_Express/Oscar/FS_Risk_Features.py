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
file_key_1 = 'AmericanExpress/Risk_Features_Imputed.csv'
file_key_2 = 'AmericanExpress/train_labels.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Reading data-files
risk_data = pd.read_csv(file_content_stream_1)
labels = pd.read_csv(file_content_stream_2)

## Removing features with inf
data = risk_data.drop(columns = ['customer_ID'], axis = 1)
to_remove = data.columns.to_series()[np.isinf(data).any()]
risk_data = risk_data.drop(columns = to_remove.index, axis = 1)

## Removing R_27 features
current_features = risk_data.columns
D_27_features = [x for x in current_features if x.startswith('R_27')]
risk_data = risk_data.drop(columns = D_27_features, axis = 1)

## Appending lables
risk_data = pd.merge(risk_data, labels, on = 'customer_ID', how = 'left')

## Defining input and target variables
X = risk_data.drop(columns = ['customer_ID', 'target'], axis = 1)
Y = risk_data['target']

## Spliting the data into train, validation, and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y)

## Defining the customized scoring function 
amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

## Defining list to store results
features_to_select = list()

for i in tqdm(range(0, 10)):
    
    ## Running RFE with Random forest
    RF_auto_feature = RFECV(estimator = RandomForestClassifier(n_estimators = 50, max_depth = 3), step = 50, scoring = amex_function, min_features_to_select = 10, cv = 3, n_jobs = -1).fit(X_train, Y_train)

    ## Appending results 
    features_to_select.append(X_train.columns[RF_auto_feature.support_])
    
## Putting results as data-frame
features_to_select = pd.DataFrame(features_to_select)
features_to_select.to_csv('Risk_Features_to_select.csv', index = False)