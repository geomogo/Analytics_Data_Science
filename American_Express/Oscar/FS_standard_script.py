import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
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

## Defining list to store results
features_to_select = list()

for i in range(0, 10):
    
    ## Running RFE with Random forest
    RF_auto_feature = RFECV(estimator = RandomForestClassifier(n_estimators = 300, max_depth = 3), step = 1, scoring = amex_function, min_features_to_select = 10, cv = 3).fit(X_train, Y_train)

    ## Appending results 
    features_to_select.append(X_train.columns[RF_auto_feature.support_])
    
## Putting results as data-frame
features_to_select = pd.DataFrame(features_to_select)