import boto3
import pandas as pd; pd.set_option('display.max_columns', 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

s3 = boto3.resource('s3')
bucket_name = 'gabriel-data-science'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'AmericanExpress/train_data.csv'
file_key_2 = 'AmericanExpress/train_labels.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Creating data-type dictionary for reading the train data-frame
dtype_dict = {'customer_ID': "object", 'S_2': "object", 'P_2': 'float16', 'B_1': 'float16', 'B_1': 'float16','B_2': 'float16',
              'R_1': 'float16','S_3': 'float16','B_41': 'float16','B_3': 'float16','B_42': 'float16','B_43': 'float16','B_44': 'float16',
              'B_4': 'float16','B_45': 'float16','B_5': 'float16','R_2': 'float16','B_46': 'float16','B_47': 'float16','B_48': 'float16',
              'B_49': 'float16','B_6': 'float16','B_7': 'float16','B_8': 'float16','B_50': 'float16','B_51': 'float16','B_9': 'float16',
              'R_3': 'float16','B_52': 'float16','P_3': 'float16','B_10': 'float16','B_53': 'float16','S_5': 'float16','B_11': 'float16',
              'S_6': 'float16','B_54': 'float16','R_4': 'float16','S_7': 'float16','B_12': 'float16','S_8': 'float16','B_55': 'float16',
              'B_56': 'float16','B_13': 'float16','R_5': 'float16','B_58': 'float16','S_9': 'float16','B_14': 'float16','B_59': 'float16',
              'B_60': 'float16','B_61': 'float16','B_15': 'float16','S_11': 'float16','B_62': 'float16','B_63': 'object','B_64': 'object',
              'B_2': 'float16','B_16': 'float16','B_17': 'float16','B_18': 'float16','B_19': 'float16','B_66': 'float16','B_20': 'float16',
              'B_68': 'float16','S_12': 'float16','R_6': 'float16','S_13': 'float16','B_21': 'float16','B_69': 'float16','B_22': 'float16',
              'B_70': 'float16','B_71': 'float16','B_72': 'float16','S_15': 'float16','B_23': 'float16','B_73': 'float16','P_4': 'float16',
              'B_74': 'float16','B_75': 'float16','B_76': 'float16','B_24': 'float16','R_7': 'float16','B_77': 'float16','B_25': 'float16',
              'B_26': 'float16','B_78': 'float16','B_79': 'float16','R_8': 'float16','R_9': 'float16','S_16': 'float16','B_80': 'float16',
              'R_10': 'float16','R_11': 'float16','B_27': 'float16','B_81': 'float16','B_82': 'float16','S_17': 'float16','R_12': 'float16',
              'B_28': 'float16','R_13': 'float16','B_83': 'float16','R_14': 'float16','R_15': 'float16','B_84': 'float16','R_16': 'float16',
              'B_29': 'float16','B_30': 'float16','S_18': 'float16','B_86': 'float16','B_87': 'float16','R_17': 'float16','R_18': 'float16',
              'B_88': 'float16','B_31': 'int64','S_19': 'float16','R_19': 'float16','B_32': 'float16','S_20': 'float16','R_20': 'float16',
              'R_21': 'float16','B_33': 'float16','B_89': 'float16','R_22': 'float16','R_23': 'float16','B_91': 'float16','B_92': 'float16',
              'B_93': 'float16','B_94': 'float16','R_24': 'float16','R_25': 'float16','B_96': 'float16','S_22': 'float16','S_23': 'float16',
              'S_24': 'float16','S_25': 'float16','S_26': 'float16','B_102': 'float16','B_103': 'float16','B_104': 'float16','B_105': 'float16',
              'B_106': 'float16','B_107': 'float16','B_36': 'float16','B_37': 'float16', 'R_26': 'float16','R_27': 'float16','B_38': 'float16',
              'B_108': 'float16','B_109': 'float16','B_110': 'float16','B_111': 'float16','B_39': 'float16','B_112': 'float16','B_40': 'float16',
              'S_27': 'float16','B_113': 'float16','B_114': 'float16','B_115': 'float16','B_116': 'float16','B_117': 'float16','B_118': 'float16',
              'B_119': 'float16','B_120': 'float16','B_121': 'float16','B_122': 'float16','B_123': 'float16','B_124': 'float16','B_125': 'float16',
              'B_126': 'float16','B_127': 'float16','B_128': 'float16','B_129': 'float16','B_41': 'float16','B_42': 'float16','B_130': 'float16',
              'B_131': 'float16','B_132': 'float16','B_133': 'float16','R_28': 'float16','B_134': 'float16','B_135': 'float16','B_136': 'float16',
              'B_137': 'float16','B_138': 'float16','B_139': 'float16','B_140': 'float16','B_141': 'float16','B_142': 'float16','B_143': 'float16',
              'B_144': 'float16','B_1': 'float16'}


## Reading data-files
train = pd.read_csv(file_content_stream_1, dtype = dtype_dict)
target = pd.read_csv(file_content_stream_2)

## Appending target variables
train = pd.merge(train, target, on = 'customer_ID', how = 'left')

## Selecting Balance variables
my_variables = train.columns
B_variables = [x for x in my_variables if x.startswith('B_')]
to_select = ['customer_ID', 'target']
for i in range(2, (len(B_variables) + 2)):
    to_select.append(B_variables[i-2])

# Balance features
balance_features = train[to_select]

## Selecting unique customer_ID and target
customer_target = train[['customer_ID', 'target']].drop_duplicates().reset_index(drop = True)

# Defining function to apply basic statistics methods
def summarize_stats(x):  
    
    b = {}
    b['B_1_mean'] = x['B_1'].mean()
    b['B_1_median'] = x['B_1'].median()
    b['B_1_min'] = x['B_1'].min()
    b['B_1_max'] = x['B_1'].max()
    b['B_1_range'] = np.where(x['B_1'].shape[0] == 1, 0, x['B_1'].max() - x['B_1'].min())
    b['B_1_IQR'] = np.where(x['B_1'].shape[0] == 1, 0,np.percentile(x['B_1'], 75) - np.percentile(x['B_1'], 25))
    b['B_1_std'] = np.where(x['B_1'].shape[0] == 1, 0, np.std(x['B_1'], ddof = 1))
#     b['B_1_negative_count'] = np.sum(x['B_1'] < 0) 
#     b['B_1_positive_count'] = np.sum(x['B_1'] > 0)
    b['B_1_pct_values_above_mean'] = np.where(x['B_1'].shape[0] == 1, 0, np.sum(x['B_1'] > x['B_1'].mean())/x['B_1'].shape[0])
    b['B_1_avg_pct_change'] = np.where(x['B_1'].shape[0] == 1, 0, pd.Series(x['B_1'].to_list()).pct_change().mean())
    
    #returning values
    return pd.Series(d, index = ['B_1_mean', 'B_1_median', 'B_1_min', 'B_1_max', 'B_1_range', 'B_1_IQR', 'B_1_std', 'B_1_pct_values_above_mean', 'B_1_avg_pct_change'])

# Applying function to dataset
data_out = balance_features.groupby('customer_ID').apply(summary_stats)
data_out['customer_ID'] = data_out.index
data_out = data_out.reset_index(drop = True)

## Joining the to datasets
data_out = pd.merge(customer_target, data_out, on = 'customer_ID', how = 'left')
data_out = data_out.drop(columns = ['target'], axis = 1)

# Reading previous balance features
balance_features = pd.read_csv('balance_features.csv')

# Aggregating new features to the dataset 
balance_features = pd.merge(balance_features, data_out, on = 'customer_ID', how = 'left')


# Exporting as csv
balance_features.to_csv('balance_features.csv', index = False)