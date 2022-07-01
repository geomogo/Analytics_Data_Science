import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from Amex_Metric import amex_metric

## Reading data-file 
data = pd.read_csv('Delinquency_Features.csv')

## Defining input and target variables
X = data.drop(columns = ['customer_ID', 'target'], axis = 1)
Y = data['target']

## Spliting the data into train, validation, and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

## Defining the customized scoring function 
amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

## Running RFE with Random forest
RF_auto_feature = RFECV(estimator = RandomForestClassifier(n_estimators = 100, max_depth = 3), step = 1, scoring = amex_function, min_features_to_select = 5, cv = 3).fit(X_train, Y_train)