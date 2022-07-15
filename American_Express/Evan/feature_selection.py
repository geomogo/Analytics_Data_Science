## Function for Payment and Spend feature selections

## Importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from Amex_Metric import amex_metric

## -------------------------------------------

## Reading customer-level data with engineered Payment and Spend features
train = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_train_payment_spend.csv')
#test = pd.read_csv('/home/ec2-user/SageMaker/Analytics_Data_Science/American_Express/Evan/amex_test_payment_spend.csv')

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
    rf_rfe = RFECV(estimator = RandomForestClassifier(max_depth = 3, n_estimators = 100), n_features_to_select = 10).fit(X_train, Y_train)
    
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








# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_selection import RFECV
# from sklearn.metrics import make_scorer
# from sklearn.ensemble import RandomForestClassifier
# from Amex_Metric import amex_metric

# ## Reading data-file 
# data = pd.read_csv('Delinquency_Features.csv')

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