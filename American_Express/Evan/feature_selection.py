## Function for Payment and Spend feature selections

## Importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

## -------------------------------------------

## Reading customer-level data with engineered Payment and Spend features
train = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_train_payment_spend.csv')
test = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_test_payment_spend.csv')

## -------------------------------------------

## Cleaning the customer-level data-frames

train = train.drop(columns = [('customer_ID', '')])

new_names = ['customer_ID']

for i in range(1, len(train.columns) - 1):
    
    to_add = new.columns[i][0] + '_' + new.columns[i][1]
    new_names.append(to_add)

new_names.append('target')

train.columns = new_names

## -------------------------------------------

## Defining the input and target variables
X = train.drop(columns = ['target'])
Y = train['target']

## Defining empty list to store results
results = list()

## Repeating steps 100 times:
for i in tqdm(range(0, 5)):
    
    ## Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.8, stratify = Y)
    
    ## Runing RFE with Random Forest as a base algorithm (with n_features_to_select = 10)
    rf_rfe = RFE(estimator = RandomForestClassifier(max_depth = 3, n_estimators = 100), n_features_to_select = 10).fit(X_train, Y_train)
    
    ## Appending the features to be selected
    results.append(rf_rfe.support_)
    
## -------------------------------------------

## Changing result lists to data-frames
results = pd.DataFrame(results, columns = X.columns)
results_df = 100 * results.apply(np.sum, axis = 0) / results.shape[0]

## Producing the final output data-frame
output = pd.DataFrame(results_df).reset_index(drop = False)
output.columns = ['Variable', 'Selected']
output = output.sort_values(by = 'Selected', ascending = False).reset_index(drop = True)
output.to_csv('feature_selection_results.csv', ignore_index = True)

## Producing the 10 feature to select
to_select = list(new_out.iloc[0:10, 0])
to_select.to_csv('feature_selection_results_top_10.csv', ignore_index = True)