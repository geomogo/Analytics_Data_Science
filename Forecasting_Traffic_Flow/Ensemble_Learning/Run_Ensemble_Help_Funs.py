import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def Run_Ensemble(train, test, model):
    
    """
    This function loops through all the directions and locations.
    With the goal of ensemble the prediction from the four
    different models that were built. It takes three arguments:
    train: this is the train data-frame.
    test: this is the test data-frame. 
    model: model to be used to ensemble the predictions 
    """
    
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
        results = Run_Ensemble_help(temp_train, temp_test, model)
        results_all_directions_val.append(results[0])
        results_all_directions_test.append(results[1])
            
    return [pd.concat(results_all_directions_val), pd.concat(results_all_directions_test)]
    
    
def Run_Ensemble_help(train, test, model):
    
    