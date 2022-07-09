## Function for variable engineering at the customer level

## Importing libraries
import pandas as pd

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
        
    ## Returning the temp data-frame
    return temp