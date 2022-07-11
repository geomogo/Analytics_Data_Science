import boto3
import pandas as pd
import numpy as np

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
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
dtype_dict = {'customer_ID': "object", 'S_2': "object", 'P_2': 'float16', 'D_39': 'float16', 'B_1': 'float16','B_2': 'float16',
              'R_1': 'float16','S_3': 'float16','D_41': 'float16','B_3': 'float16','D_42': 'float16','D_43': 'float16','D_44': 'float16',
              'B_4': 'float16','D_45': 'float16','B_5': 'float16','R_2': 'float16','D_46': 'float16','D_47': 'float16','D_48': 'float16',
              'D_49': 'float16','B_6': 'float16','B_7': 'float16','B_8': 'float16','D_50': 'float16','D_51': 'float16','B_9': 'float16',
              'R_3': 'float16','D_52': 'float16','P_3': 'float16','B_10': 'float16','D_53': 'float16','S_5': 'float16','B_11': 'float16',
              'S_6': 'float16','D_54': 'float16','R_4': 'float16','S_7': 'float16','B_12': 'float16','S_8': 'float16','D_55': 'float16',
              'D_56': 'float16','B_13': 'float16','R_5': 'float16','D_58': 'float16','S_9': 'float16','B_14': 'float16','D_59': 'float16',
              'D_60': 'float16','D_61': 'float16','B_15': 'float16','S_11': 'float16','D_62': 'float16','D_63': 'object','D_64': 'object',
              'D_65': 'float16','B_16': 'float16','B_17': 'float16','B_18': 'float16','B_19': 'float16','D_66': 'float16','B_20': 'float16',
              'D_68': 'float16','S_12': 'float16','R_6': 'float16','S_13': 'float16','B_21': 'float16','D_69': 'float16','B_22': 'float16',
              'D_70': 'float16','D_71': 'float16','D_72': 'float16','S_15': 'float16','B_23': 'float16','D_73': 'float16','P_4': 'float16',
              'D_74': 'float16','D_75': 'float16','D_76': 'float16','B_24': 'float16','R_7': 'float16','D_77': 'float16','B_25': 'float16',
              'B_26': 'float16','D_78': 'float16','D_79': 'float16','R_8': 'float16','R_9': 'float16','S_16': 'float16','D_80': 'float16',
              'R_10': 'float16','R_11': 'float16','B_27': 'float16','D_81': 'float16','D_82': 'float16','S_17': 'float16','R_12': 'float16',
              'B_28': 'float16','R_13': 'float16','D_83': 'float16','R_14': 'float16','R_15': 'float16','D_84': 'float16','R_16': 'float16',
              'B_29': 'float16','B_30': 'float16','S_18': 'float16','D_86': 'float16','D_87': 'float16','R_17': 'float16','R_18': 'float16',
              'D_88': 'float16','B_31': 'int64','S_19': 'float16','R_19': 'float16','B_32': 'float16','S_20': 'float16','R_20': 'float16',
              'R_21': 'float16','B_33': 'float16','D_89': 'float16','R_22': 'float16','R_23': 'float16','D_91': 'float16','D_92': 'float16',
              'D_93': 'float16','D_94': 'float16','R_24': 'float16','R_25': 'float16','D_96': 'float16','S_22': 'float16','S_23': 'float16',
              'S_24': 'float16','S_25': 'float16','S_26': 'float16','D_102': 'float16','D_103': 'float16','D_104': 'float16','D_105': 'float16',
              'D_106': 'float16','D_107': 'float16','B_36': 'float16','B_37': 'float16', 'R_26': 'float16','R_27': 'float16','B_38': 'float16',
              'D_108': 'float16','D_109': 'float16','D_110': 'float16','D_111': 'float16','B_39': 'float16','D_112': 'float16','B_40': 'float16',
              'S_27': 'float16','D_113': 'float16','D_114': 'float16','D_115': 'float16','D_116': 'float16','D_117': 'float16','D_118': 'float16',
              'D_119': 'float16','D_120': 'float16','D_121': 'float16','D_122': 'float16','D_123': 'float16','D_124': 'float16','D_125': 'float16',
              'D_126': 'float16','D_127': 'float16','D_128': 'float16','D_129': 'float16','B_41': 'float16','B_42': 'float16','D_130': 'float16',
              'D_131': 'float16','D_132': 'float16','D_133': 'float16','R_28': 'float16','D_134': 'float16','D_135': 'float16','D_136': 'float16',
              'D_137': 'float16','D_138': 'float16','D_139': 'float16','D_140': 'float16','D_141': 'float16','D_142': 'float16','D_143': 'float16',
              'D_144': 'float16','D_145': 'float16'}

## Reading data-files
train = pd.read_csv(file_content_stream_1, dtype = dtype_dict)
target = pd.read_csv(file_content_stream_2)

delinquency_features = pd.read_csv('Delinquency_Features.csv')

## Appending target variables
train = pd.merge(train, target, on = 'customer_ID', how = 'left')

## Selecting Deliquency variables
my_variables = train.columns
D_variables = [x for x in my_variables if x.startswith('D_')]
to_select = ['customer_ID', 'target']
for i in range(2, (len(D_variables) + 2)):
    to_select.append(D_variables[i-2])

train_deli = train[to_select]

## Selecting unique customer_ID and target
customer_target = train[['customer_ID', 'target']].drop_duplicates().reset_index(drop = True)

## Computing basic summary-stats features
def summary_stats(x):
    
    d = {}
    d['D_89_mean'] = x['D_89'].mean()
    d['D_89_median'] = x['D_89'].median()
    d['D_89_min'] = x['D_89'].min()
    d['D_89_max'] = x['D_89'].max()
    d['D_89_range'] = np.where(x['D_89'].shape[0] == 1, 0, x['D_89'].max() - x['D_89'].min())
    d['D_89_IQR'] = np.where(x['D_89'].shape[0] == 1, 0,np.percentile(x['D_89'], 75) - np.percentile(x['D_89'], 25))
    d['D_89_std'] = np.where(x['D_89'].shape[0] == 1, 0, np.std(x['D_89'], ddof = 1))
#     d['D_89_negative_count'] = np.sum(x['D_89'] < 0) 
#     d['D_89_positive_count'] = np.sum(x['D_89'] > 0)
    d['D_89_pct_values_above_mean'] = np.where(x['D_89'].shape[0] == 1, 0, np.sum(x['D_89'] > x['D_89'].mean())/x['D_89'].shape[0])
    d['D_89_avg_pct_change'] = np.where(x['D_89'].shape[0] == 1, 0, pd.Series(x['D_89'].to_list()).pct_change().mean())
    
    return pd.Series(d, index = ['D_89_mean', 'D_89_median', 'D_89_min', 'D_89_max', 'D_89_range', 'D_89_IQR', 'D_89_std', 'D_89_pct_values_above_mean', 'D_89_avg_pct_change'])

data_out = train_deli.groupby('customer_ID').apply(summary_stats)
data_out['customer_ID'] = data_out.index
data_out = data_out.reset_index(drop = True)

# ## Computing average change at the customer level
# data_change = pd.DataFrame(train_deli.groupby(['customer_ID'])['D_89'].apply(lambda x: pd.Series(x.to_list()).pct_change().mean()))
# data_change['customer_ID'] = data_change.index
# data_change = data_change.reset_index(drop = True)
# data_change.columns = ['D_89_change', 'customer_ID']

# ## Computing change from first to last month
# data_change_first_last = pd.DataFrame(train_deli.groupby(['customer_ID'])['D_89'].apply(lambda x: pd.Series(x.iloc[[0, -1]].to_list()).pct_change())).unstack()
# data_change_first_last = data_change_first_last.drop(columns = ('D_89', 0), axis = 1)
# data_change_first_last['customer_ID'] = data_change_first_last.index
# data_change_first_last = data_change_first_last.reset_index(drop = True)
# data_change_first_last.columns = ['D_89_change_first_last', 'customer_ID']

## Joining the to datasets
data_out = pd.merge(customer_target, data_out, on = 'customer_ID', how = 'left')
data_out = data_out.drop(columns = ['target'], axis = 1)

delinquency_features = pd.merge(delinquency_features, data_out, on = 'customer_ID', how = 'left')

# data_out = pd.merge(data_avg, data_median, on = 'customer_ID', how = 'left')
# data_out = pd.merge(data_out, data_change, on = 'customer_ID', how = 'left')
# data_out = pd.merge(data_out, data_change_first_last, on = 'customer_ID', how = 'left')

# data_out.to_csv('Delinquency_Features.csv', index = False)
delinquency_features.to_csv('Delinquency_Features.csv', index = False)
