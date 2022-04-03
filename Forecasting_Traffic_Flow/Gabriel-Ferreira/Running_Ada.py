import boto3
import pandas as pd 
import numpy as np
from Running_AdaBoost_Help_Funs import main_ADA_all_directions

## Importing train and test data
train = pd.read_csv('/Users/gabrielvictorgomesferreira/Desktop/DataCompetitions/TabularPlayground/train.csv')
test = pd.read_csv('/Users/gabrielvictorgomesferreira/Desktop/DataCompetitions/TabularPlayground/test.csv')

## Puting time in the right format 
train['time'] = pd.to_datetime(train['time'], format = '%Y-%m-%d %H:%M:%S')
test['time'] = pd.to_datetime(test['time'], format = '%Y-%m-%d %H:%M:%S')

## Extracting day, hour and minute
train['day'] = train['time'].dt.dayofweek
train['hour'] = train['time'].dt.hour
train['minute'] = train['time'].dt.minute

test['day'] = test['time'].dt.dayofweek
test['hour'] = test['time'].dt.hour
test['minute'] = test['time'].dt.minute

## Changing direction to dummies
train = pd.concat([train, pd.get_dummies(train['direction'])], axis = 1)
test = pd.concat([test, pd.get_dummies(train['direction'])], axis = 1)

## Modeling 
results = main_ADA_all_directions(train, test)

## Storing results
results[0].to_csv('results_validation.csv', index = False)
results[1].to_csv('results_test.csv', index = False)