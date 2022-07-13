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
#test = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_test_payment_spend.csv')

## -------------------------------------------

## Cleaning the customer-level data-frames
train = train.drop(columns = ["('customer_ID', '')"])
train.columns = ['customer_ID', 'P_2_mean', 'P_2_median', 'P_2_sum', 'P_2_count', 'P_2_data_range', 'P_2_iqr', 'P_2_avg_pct_change', 'P_2_correlation', 'P_3_mean', 'P_3_median', 'P_3_sum', 'P_3_count', 'P_3_data_range', 'P_3_iqr', 'P_3_avg_pct_change', 'P_3_correlation', 'P_4_mean', 'P_4_median', 'P_4_sum', 'P_4_data_range', 'P_4_iqr', 'P_4_avg_pct_change', 'P_4_correlation', 'S_3_median', 'S_3_sum', 'S_3_data_range', 'S_3_iqr', 'S_3_avg_pct_change', 'S_3_correlation', 'S_5_data_range', 'S_5_iqr', 'S_5_avg_pct_change', 'S_5_correlation', 'S_6_mean', 'S_6_median', 'S_6_sum', 'S_6_mad', 'S_6_data_range', 'S_6_iqr', 'S_6_avg_pct_change', 'S_6_correlation', 'S_8_mean', 'S_8_median', 'S_8_sum', 'S_8_data_range', 'S_8_iqr', 'S_8_avg_pct_change', 'S_8_correlation', 'S_13_mean', 'S_13_sum', 'S_13_std', 'S_13_mad', 'S_13_data_range', 'S_13_iqr', 'S_13_avg_pct_change', 'S_13_correlation', 'S_25_mean', 'S_25_sum', 'S_25_std', 'S_25_mad', 'S_25_count', 'S_25_data_range', 'S_25_iqr', 'S_25_avg_pct_change', 'S_25_correlation', 'S_27_count', 'S_27_data_range', 'S_27_iqr', 'S_27_avg_pct_change', 'S_27_correlation', 'target']

# train = train.dropna()
# Need to drop missing values

## -------------------------------------------

## Defining the input and target variables
X = train.drop(columns = ['customer_ID', 'target'])
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
output.to_csv('feature_selection_results.csv', index = False)

## Producing the 10 feature to select
to_select = output.iloc[0:10, 0]
to_select.to_csv('feature_selection_results_top_10.csv', index = False)