## Function for Payment and Spend feature selections

## Importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import miceforest as mf
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from Amex_Metric import amex_metric

## -------------------------------------------

## Sanity check
print('-- Feature Selection Process Started --')

## Reading customer-level data with engineered Payment and Spend features
train = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_train_payment_spend.csv')
#test = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_test_payment_spend.csv')

## Sanity check
print('-- Data Read --')

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

## Subsetting the data for the 24 most important variables
train_impute = train_impute[['customer_ID', 'P_2_mean', 'S_3_median', 'S_25_iqr', 'S_25_data_range', 'S_25_mad', 'S_25_std', 'S_25_sum', 'S_25_mean', 'S_13_sum', 'S_13_mean', 'S_8_sum', 'P_2_median', 'S_8_mean', 'S_5_correlation', 'S_3_sum', 'S_8_median', 'P_2_sum', 'P_2_data_range', 'P_3_mean', 'P_3_sum', 'P_2_correlation', 'P_4_mean', 'P_3_median', 'P_4_sum', 'target']]

## Sanity check
print('-- Data subsetting complete -- \n')

## Defining the input and target variables
X = train_impute.drop(columns = ['customer_ID', 'target'])
Y = train_impute['target']

## Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y)

## Defining the customized scoring function 
amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

## Defining empty list to store results
features_to_select = list()

## Repeating RFECV steps 10 times:
for i in tqdm(range(0, 10)):
    
    ## Runing RFECV with Random Forest as a base algorithm
    rf_rfecv = RFECV(estimator = RandomForestClassifier(n_estimators = 100, max_depth = 3), step = 2, scoring = amex_function, min_features_to_select = 2, cv = 3).fit(X_train, Y_train)
    
    ## Appending results 
    features_to_select.append(rf_rfecv.support_)
    
## Creating a data-frame to stre results
features_to_select = pd.DataFrame(features_to_select, columns = X.columns)
features_to_select = 100 * features_to_select.apply(np.sum, axis = 0) / features_to_select.shape[0]

## Producing the final output data-frame
output = pd.DataFrame(features_to_select).reset_index(drop = False)
output.columns = ['Variable', 'Selected']
output = output.sort_values(by = 'Selected', ascending = False).reset_index(drop = True)
output.to_csv('feature_selection_results_pt2.csv', index = False)

## Sanity check
print('-- Completed --')
    
## -------------------------------------------