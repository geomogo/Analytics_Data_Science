import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor


def main_XGB_all_directions(data):
    
    '''
    This function
    '''
    
    ## Defining all the directions
    directions = data['direction'].unique()
    
    ## Defining list to store results
    results_all_directions = list()
    
    for i in range(0, len(directions)):
        
        ## Subsetting data based on directions
        temp_data = data[data['direction'] == directions[i]].reset_index(drop = True)
        
        ## Appending results 
        results_all_directions.append(main_XGB_all_directions_help(temp_data))
        
        
def main_XGB_all_directions_help(data):
        
    ## Defining locations 
    x_values = data['x'].unique()
    y_values = data['y'].unique()
    
    ## Defining list to store results
    results_all_locations = list()
    
    for i in range(0, len(x_values)):
        
        for j in range(0, len(y_values)):
            
            temp_data = data[(data['x'] == x_values[i]) & (data['y'] == y_values[j])].reset_index(drop = True)
            
            

def main_XGB_all_directions_help_help(data):            
    
    ## Defining train & validation datasets
    X_train = data.loc[0:13023, ['day', 'hour', 'minute']]
    Y_train = data.loc[0:13023, ['congestion']]

    X_val = data.loc[13023:13059, ['day', 'hour', 'minute']]
    Y_val = data.loc[13023:13059, ['congestion']]
    
    ## Defining the hyper-parameter grid
    XGBoost_param_grid = {'n_estimators': [300],
                          'max_depth': [5, 7],
                          'min_child_weight': [5, 7, 10],
                          'learning_rate': [0.01],
                          'gamma': [0.3, 0.1],
                          'subsample': [0.8, 1],
                          'colsample_bytree': [1]}

    ## Performing grid search with 5 folds
    XGBoost_grid_search = GridSearchCV(XGBRegressor(), XGBoost_param_grid, cv = 5, scoring = 'neg_mean_squared_error').fit(X_train, Y_train)

    ## Extracting the best model
    XGBoost_md = XGBoost_grid_search.best_estimator_

    ## Predicting on validation 
    XGBoost_pred = XGBoost_md.predict(X_val)
            
            
            