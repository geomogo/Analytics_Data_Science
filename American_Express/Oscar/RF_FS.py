import boto3
import pandas as pd; pd.set_option('display.max_columns', 500)
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'AmericanExpress/Delinquency_Features_Filled.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading data-files
data = pd.read_csv(file_content_stream)
data = data.drop(columns = ['D_64_last'], axis = 1)


def pandas_to_list(data):
    
    results = []
    n = data.shape[0]
    
    for i in range(0, n):
        
        a = data.loc[0].values.tolist()
        a = [x for x in a if str(x) != 'nan']
        results.append(a)
    
    results = [i for ii in results for i in ii]
    return results

## Reading data-files
data0 = pd.read_csv('Delinquency_Features_to_select.csv')
data1 = pd.read_csv('Delinquency_Features_to_select_1.csv')
data2 = pd.read_csv('Delinquency_Features_to_select_2.csv')
data3 = pd.read_csv('Delinquency_Features_to_select_3.csv')
data4 = pd.read_csv('Delinquency_Features_to_select_4.csv')
data5 = pd.read_csv('Delinquency_Features_to_select_5.csv')
data6 = pd.read_csv('Delinquency_Features_to_select_6.csv')
data7 = pd.read_csv('Delinquency_Features_to_select_7.csv')
data8 = pd.read_csv('Delinquency_Features_to_select_8.csv')
data9 = pd.read_csv('Delinquency_Features_to_select_9.csv')

x0 = pandas_to_list(data0)
x1 = pandas_to_list(data1)
x2 = pandas_to_list(data2)
x3 = pandas_to_list(data3)
x4 = pandas_to_list(data4)
x5 = pandas_to_list(data5)
x6 = pandas_to_list(data6)
x7 = pandas_to_list(data7)
x8 = pandas_to_list(data8)
x9 = pandas_to_list(data9)

## Combining all list
a = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9

## Converting to data-frame
features_rank = pd.DataFrame(a)
feature_freq = 100*features_rank[0].value_counts().sort_values(ascending = False) / features_rank.shape[0]
feature_names = features_rank[0].value_counts().sort_values(ascending = False).index.tolist()
features_rank = pd.DataFrame({'feature': feature_names, 'freq': feature_freq}).reset_index(drop = True)
features_rank = features_rank[features_rank['freq'] > 0.22]

## Putting variables in the right shape 
data['D_68_last'] = data['D_68_last'].astype(str)
data['D_114_last'] = data['D_114_last'].astype(str)
data['D_116_last'] = data['D_116_last'].astype(str)
data['D_117_last'] = data['D_117_last'].astype(str)
data['D_120_last'] = data['D_120_last'].astype(str)
data['D_126_last'] = data['D_126_last'].astype(str)

## Converting to dummies
dummies = pd.get_dummies(data[['D_63_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last', 'D_126_last']])

## Appeding dummies 
data = data.drop(columns = ['D_63_last', 'D_68_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last', 'D_126_last'], axis = 1)
data = pd.concat([data, dummies], axis = 1)

## Defining input and target variables
X = data[features_rank['feature'].tolist()]
Y = data['target']

## Defining list to store results
results = []

for i in tqdm(range(0, 10)):
    
    ## Spliting the data into train, validation, and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y)
    
    ## Building the model
    RF_md = RandomForestClassifier(n_estimators = 100, max_depth = 3).fit(X_train, Y_train)
    
    results.append(RF_md.feature_importances_)
    
## Defining data-frame to be stored
results = pd.DataFrame(results)
results = results.apply(np.mean, axis = 0)
feature_imp = pd.DataFrame({'feature': features_rank['feature'].tolist(), 'imp': results})
feature_imp.to_csv('RF_Importance.csv', index = False)