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
    
    