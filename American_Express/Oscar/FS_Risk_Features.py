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
file_key_1 = 'AmericanExpress/Risk_Features.csv'
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

