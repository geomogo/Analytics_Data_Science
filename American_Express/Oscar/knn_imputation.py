import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

## Reading data-file
delinquency_data = pd.read_csv('Delinquency_Features.csv')

## Indentify variables with nan


## Running KNNImputer