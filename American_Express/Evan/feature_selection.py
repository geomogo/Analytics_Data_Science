## Function for Payment and Spend feature selections

## Importing libraries
import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from Amex_Metric import amex_metric

## -------------------------------------------

## Sanity check
print('-- Process Started --')

## Reading customer-level data with engineered Payment and Spend features
train = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_train_payment_spend.csv')
#test = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_test_payment_spend.csv')

## -------------------------------------------

## Imputing the train data-frame using the Mice Forest library

## Defining the input variables and dropping categorical variables
mf_train = train.drop(columns = ['customer_ID', 'target'])

# Building the miceforest kernel
kernel_train = mf.ImputationKernel(mf_train, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
train_impute = kernel_train.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
train_impute = pd.concat([train[['customer_ID', 'target']], train_impute], axis = 1)

## Sanity check
print('-- Training data-frame imputation complete -- \n')

## -------------------------------------------

## Defining the input and target variables
X = train_impute.drop(columns = ['customer_ID', 'target'])
Y = train_impute['target']

## Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

## Defining the customized scoring function 
amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

## Defining empty list to store results
features_to_select = list()

## Repeating steps 100 times:
for i in range(0, 5):
    
    ## Runing RFECV with Random Forest as a base algorithm
    rf_rfecv = RFECV(estimator = RandomForestClassifier(n_estimators = 300, max_depth = 3), step = 1, scoring = amex_function, min_features_to_select = 5, cv = 3).fit(X_train, Y_train)
    
    ## Appending results 
    features_to_select.append(X_train.columns[rf_rfecv.support_])
    
## Putting results as data-frame
features_to_select = pd.DataFrame(features_to_select)

features_to_select.to_csv('feature_selection_results.csv', index = False)

## Sanity check
print('-- Completed --')
    
## -------------------------------------------

# ## Changing result lists to data-frames
# results = pd.DataFrame(results, columns = X.columns)
# results_df = 100 * results.apply(np.sum, axis = 0) / results.shape[0]

# ## Producing the final output data-frame
# output = pd.DataFrame(results_df).reset_index(drop = False)
# output.columns = ['Variable', 'Selected']
# output = output.sort_values(by = 'Selected', ascending = False).reset_index(drop = True)
# output.to_csv('feature_selection_results.csv', index = False)

# ## Producing the 10 feature to select
# to_select = output.iloc[0:10, 0]
# to_select.to_csv('feature_selection_results_top_10.csv', index = False)



# ## Defining input and target variables
# X = data.drop(columns = ['customer_ID', 'target'], axis = 1)
# Y = data['target']

# ## Spliting the data into train, validation, and test
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

# ## Defining the customized scoring function 
# amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

# ## Defining list to store results
# features_to_select = list()

# for i in range(0, 10):
    
#     ## Running RFE with Random forest
#     RF_auto_feature = RFECV(estimator = RandomForestClassifier(n_estimators = 300, max_depth = 3), step = 1, scoring = amex_function, min_features_to_select = 10, cv = 3).fit(X_train, Y_train)

#     ## Appending results 
#     features_to_select.append(X_train.columns[RF_auto_feature.support_])
    
# ## Putting results as data-frame
# features_to_select = pd.DataFrame(features_to_select)