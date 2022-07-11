## Function for variable engineering at the customer level

## Importing libraries
import pandas as pd
import numpy as np

## Defining the "create_var" function
def create_var(data, variable, method, new_name):
    
    ## Calculating the mean
    if (method == 'mean'):
        
        ## Grouping the data by customer_ID to obtain the mean values
        temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].mean()).reset_index(drop = False)
        
        ## Cleaning the resulting data-frame
        temp.columns = ['customer_ID', new_name]
        
    ## Calculating the median
    elif (method == 'median'):
        
        ## Grouping the data by customer_ID to obtain the mean values
        temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].median()).reset_index(drop = False)
        
        ## Cleaning the resulting data-frame
        temp.columns = ['customer_ID', new_name]
        
    ## Calculating the sum
    elif (method == 'sum'):
        
        ## Grouping the data by customer_ID to obtain the mean values
        temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].sum()).reset_index(drop = False)
        
        ## Cleaning the resulting data-frame
        temp.columns = ['customer_ID', new_name]
        
    ## Calculating the min
    elif (method == 'sum'):
        
        ## Grouping the data by customer_ID to obtain the mean values
        temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].min()).reset_index(drop = False)
        
        ## Cleaning the resulting data-frame
        temp.columns = ['customer_ID', new_name]
        
    ## Calculating the max
    elif (method == 'sum'):
        
        ## Grouping the data by customer_ID to obtain the mean values
        temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].max()).reset_index(drop = False)
        
        ## Cleaning the resulting data-frame
        temp.columns = ['customer_ID', new_name]
        
    ## Returning the temp data-frame
    return temp



## Defining the "create" function
def create(data, variable):
    
    ## Calculating the mean
    mean_temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].mean()).reset_index(drop = False)
    mean_temp.columns = ['customer_ID', variable + '_mean']
    
    ## Calculating the median
    median_temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].median()).reset_index(drop = False)
    median_temp.columns = ['customer_ID', variable + '_median']
    
    ## Calculating the sum
    sum_temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].sum()).reset_index(drop = False)
    sum_temp.columns = ['customer_ID', variable + '_sum']
    
    ## Calculating the min
    min_temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].min()).reset_index(drop = False)
    min_temp.columns = ['customer_ID', variable + '_min']
    
    ## Calculating the max
    max_temp = pd.DataFrame(data.groupby(['customer_ID'])[variable].max()).reset_index(drop = False)
    max_temp.columns = ['customer_ID', variable + '_max']
    
    ## Concatenating all feature engineering data-frames into a single object
    new_vars = pd.concat([mean_temp, median_temp.iloc[:, 1], sum_temp.iloc[:, 1], min_temp.iloc[:, 1], max_temp.iloc[:, 1]], axis = 1)
    
    ## Return statement
    return new_vars

## -------------------------------------------

## Simplifying the aggregation functions

def data_range(x):
    return x.max() - x.min()

def iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

def avg_pct_change(x):
    return pd.Series(x.to_list()).pct_change().mean()