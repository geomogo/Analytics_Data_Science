import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def main_ADA_all_directions(train, test):
    
    '''
    This function loops through all the directions and locations. 
    It takes two arguments:
    train: this is the train data-frame.
    test: this is the test data-frame.
    '''
    
    ## Defining all the directions
    directions = train['direction'].unique()
    
    ## Defining lists to store results
    results_all_directions_val = list()
    results_all_directions_test = list()
    
    for i in range(0, len(directions)):
        print('Working on direction:', directions[i])
        ## Subsetting train & test based on directions
        temp_train = train[train['direction'] == directions[i]].reset_index(drop = True)
        temp_test = test[test['direction'] == directions[i]].reset_index(drop = True)
        
        ## Appending results 
        results = main_ADA_all_directions_help(temp_train, temp_test)
        results_all_directions_val.append(results[0])
        results_all_directions_test.append(results[1])
            
    return [pd.concat(results_all_directions_val), pd.concat(results_all_directions_test)]
        
        
def main_ADA_all_directions_help(train, test):
    
    ## Defining lists to store results
    results_all_locations_val = list()
    results_all_locations_test = list()
    
    ## Defining locations 
    x_values = train['x'].unique()
    y_values = train['y'].unique()
    
    ## Defining list to store results
    results_all_locations = list()
    
    for i in range(0, len(x_values)):
        
        for j in range(0, len(y_values)):
            print('location: (',x_values[i],',',y_values[j],')')
            ## Subsetting train & test based on locaitons
            temp_train = train[(train['x'] == x_values[i]) & (train['y'] == y_values[j])].reset_index(drop = True)
            temp_test = test[(test['x'] == x_values[i]) & (test['y'] == y_values[j])].reset_index(drop = True)
            
            ## Sanity check
            if (temp_train.shape[0] == 0):
                
                continue
            
            ## Modeling building and prediction at location (x, y)
            results = main_ADA_all_directions_help_help(temp_train, temp_test)
            results_all_locations_val.append(results[0])
            results_all_locations_test.append(results[1])
            
    return [pd.concat(results_all_locations_val), pd.concat(results_all_locations_test)]
            

def main_ADA_all_directions_help_help(train, test):            
    
    ## Defining train, validation, and test datasets
    X_train = train.loc[0:13023, ['day', 'hour', 'minute']]
    Y_train = train.loc[0:13023, ['congestion']]
    Y_train = Y_train['congestion']

    X_val = train.loc[13023:13059, ['day', 'hour', 'minute']]
    Y_val = train.loc[13023:13059, ['congestion']]
    Y_val = Y_val['congestion']
    
    X_test = test[['day', 'hour', 'minute']]
    
    
    ## Defining the hyper-parameters for RF
    Ada_param_grid = {'n_estimators': [100, 300, 500],
                 'base_estimator__min_samples_split': [10, 15], 
                 'base_estimator__min_samples_leaf': [5, 7], 
                 'base_estimator__max_depth' : [3, 5, 7],
                 'learning_rate': [0.001, 0.01, 0.1]}

    ## Running grid search with 3 fold
    Ada_grid_search = GridSearchCV(AdaBoostRegressor(base_estimator = DecisionTreeRegressor()), Ada_param_grid, cv = 3, scoring = 'neg_mean_absolute_error', n_jobs = -1).fit(X_train, Y_train)

    # Extracting the best model
    ada_md = Ada_grid_search.best_estimator_


    ## Predicting on validation & test 
    ada_val_pred = ada_md.predict(X_val)
    ada_test_pred = ada_md.predict(X_test)
    
    ## Appending predictions on validation and test
    data_out = train.loc[13023:13059].reset_index(drop = True)
    data_out['congestion_pred'] = ada_val_pred
    test['congestion_pred'] = ada_test_pred

    
    return [data_out[['row_id', 'time', 'x', 'y', 'direction', 'congestion', 'congestion_pred']], test[['row_id', 'time', 'x', 'y', 'direction', 'congestion_pred']]]
     
    
