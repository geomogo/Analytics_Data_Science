## Importing libraries
import boto3
import pandas as pd
import numpy as np
import miceforest as mf

## -------------------------------------------

## Sanity check
print('-- Process Starting --')

## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'evan-callaghan-bucket'
bucket = s3.Bucket(bucket_name)

file_key = 'Kaggle-American-Express-Default/amex_train_data.csv'
file_key2 = 'Kaggle-American-Express-Default/amex_train_labels.csv'

bucket_object = bucket.Object(file_key)
bucket_object2 = bucket.Object(file_key2)

file_object = bucket_object.get()
file_object2 = bucket_object2.get()

file_content_stream = file_object.get('Body')
file_content_stream2 = file_object2.get('Body')

## Creating data-type dictionary for reading the train data-frame
dtype_dict = {'customer_ID': "object", 'S_2': "object", 'P_2': 'float16', 'D_39': 'float16', 'B_1': 'float16','B_2': 'float16', 'R_1': 'float16','S_3': 'float16','D_41': 'float16','B_3': 'float16','D_42': 'float16','D_43': 'float16','D_44': 'float16', 'B_4': 'float16','D_45': 'float16','B_5': 'float16','R_2': 'float16','D_46': 'float16','D_47': 'float16','D_48': 'float16', 'D_49': 'float16','B_6': 'float16','B_7': 'float16','B_8': 'float16','D_50': 'float16','D_51': 'float16','B_9': 'float16', 'R_3': 'float16','D_52': 'float16','P_3': 'float16','B_10': 'float16','D_53': 'float16','S_5': 'float16','B_11': 'float16', 'S_6': 'float16','D_54': 'float16','R_4': 'float16','S_7': 'float16','B_12': 'float16','S_8': 'float16','D_55': 'float16', 'D_56': 'float16','B_13': 'float16','R_5': 'float16','D_58': 'float16','S_9': 'float16','B_14': 'float16','D_59': 'float16', 'D_60': 'float16','D_61': 'float16','B_15': 'float16','S_11': 'float16','D_62': 'float16','D_63': 'object','D_64': 'object', 'D_65': 'float16','B_16': 'float16','B_17': 'float16','B_18': 'float16','B_19': 'float16','D_66': 'float16','B_20': 'float16', 'D_68': 'float16','S_12': 'float16','R_6': 'float16','S_13': 'float16','B_21': 'float16','D_69': 'float16','B_22': 'float16', 'D_70': 'float16','D_71': 'float16','D_72': 'float16','S_15': 'float16','B_23': 'float16','D_73': 'float16','P_4': 'float16', 'D_74': 'float16','D_75': 'float16','D_76': 'float16','B_24': 'float16','R_7': 'float16','D_77': 'float16','B_25': 'float16', 'B_26': 'float16','D_78': 'float16','D_79': 'float16','R_8': 'float16','R_9': 'float16','S_16': 'float16','D_80': 'float16', 'R_10': 'float16','R_11': 'float16','B_27': 'float16','D_81': 'float16','D_82': 'float16','S_17': 'float16','R_12': 'float16', 'B_28': 'float16','R_13': 'float16','D_83': 'float16','R_14': 'float16','R_15': 'float16','D_84': 'float16','R_16': 'float16', 'B_29': 'float16','B_30': 'float16','S_18': 'float16','D_86': 'float16','D_87': 'float16','R_17': 'float16','R_18': 'float16', 'D_88': 'float16','B_31': 'int64','S_19': 'float16','R_19': 'float16','B_32': 'float16','S_20': 'float16','R_20': 'float16', 'R_21': 'float16','B_33': 'float16','D_89': 'float16','R_22': 'float16','R_23': 'float16','D_91': 'float16','D_92': 'float16', 'D_93': 'float16','D_94': 'float16','R_24': 'float16','R_25': 'float16','D_96': 'float16','S_22': 'float16','S_23': 'float16', 'S_24': 'float16','S_25': 'float16','S_26': 'float16','D_102': 'float16','D_103': 'float16','D_104': 'float16','D_105': 'float16', 'D_106': 'float16','D_107': 'float16','B_36': 'float16','B_37': 'float16', 'R_26': 'float16','R_27': 'float16','B_38': 'float16', 'D_108': 'float16','D_109': 'float16','D_110': 'float16','D_111': 'float16','B_39': 'float16','D_112': 'float16','B_40': 'float16', 'S_27': 'float16','D_113': 'float16','D_114': 'float16','D_115': 'float16','D_116': 'float16','D_117': 'float16','D_118': 'float16', 'D_119': 'float16','D_120': 'float16','D_121': 'float16','D_122': 'float16','D_123': 'float16','D_124': 'float16','D_125': 'float16', 'D_126': 'float16','D_127': 'float16','D_128': 'float16','D_129': 'float16','B_41': 'float16','B_42': 'float16','D_130': 'float16', 'D_131': 'float16','D_132': 'float16','D_133': 'float16','R_28': 'float16','D_134': 'float16','D_135': 'float16','D_136': 'float16', 'D_137': 'float16','D_138': 'float16','D_139': 'float16','D_140': 'float16','D_141': 'float16','D_142': 'float16','D_143': 'float16', 'D_144': 'float16','D_145': 'float16'}

## Reading the data
train = pd.read_csv(file_content_stream, dtype = dtype_dict)
train_labels = pd.read_csv(file_content_stream2)

## Subsetting the data for Payment and Spend variables
train = train[['customer_ID', 'P_2', 'P_3', 'P_4', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 
               'S_18', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']]

## Appending target labels with train data-frame
train = train.merge(train_labels, on = 'customer_ID', how = 'left')

print(train.columns)

## Sanity check
print('-- Data Read -- \n')

## -------------------------------------------

## Computing new Payment and Spend count features with the train data-frame
count_vars_train = train.groupby('customer_ID').agg({'P_3':['count'], 'S_3':['count'], 'S_7':['count'], 
                                                     'S_22': ['count'], 'S_23': ['count'], 'S_25': ['count']}).reset_index(drop = False)

## Editing the variable names in the data-frame
count_vars_train.columns = ['customer_ID', 'P_3_count', 'S_3_count', 'S_7_count', 'S_22_count', 'S_23_count', 'S_25_count']

# Creating binary count variables
count_vars_train['P_3_count_binary'] = np.where(count_vars_train["P_3_count"] > 0, 1, 0)
count_vars_train['S_3_count_binary'] = np.where(count_vars_train["S_3_count"] > 0, 1, 0)
count_vars_train['S_7_count_binary'] = np.where(count_vars_train["S_7_count"] > 0, 1, 0)
count_vars_train['S_22_count_binary'] = np.where(count_vars_train["S_22_count"] > 0, 1, 0)
count_vars_train['S_23_count_binary'] = np.where(count_vars_train["S_23_count"] > 0, 1, 0)
count_vars_train['S_25_count_binary'] = np.where(count_vars_train["S_25_count"] > 0, 1, 0)

## Sanity check
print('-- Training counts data-frame complete -- \n')

## -------------------------------------------

## Imputing the train data-frame using the Mice Forest library

## Defining the input variables and dropping categorical variables
mf_train = train.drop(columns = ['customer_ID', 'target'])

# Building the miceforest kernel
kernel_train = mf.ImputationKernel(mf_train, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
train_impute = kernel_train.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
train_impute = pd.concat([train[['customer_ID', 'target']], train_impute], axis = 1)

## Sanity check
print('-- Training data-frame imputation complete -- \n')

## -------------------------------------------

## Creating other aggregated features using the imputed train data-frame

## Creating a series of aggregation functions
def data_range(x):
    return x.max() - x.min()
def iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)
def avg_pct_change(x):
    return pd.Series(x.to_list()).pct_change()[1:12].mean()
def correlation(x):
    return pd.Series(x.values).corr(other = pd.Series(x.index), method = 'pearson')

## Creating new Payment features with the cleaned train data-frame
payment_vars_train = train_impute.groupby('customer_ID').agg({'P_2':['mean', 'median', 'sum', data_range, iqr, avg_pct_change, correlation], 'P_3':['mean', 'median', 'sum', data_range, iqr, avg_pct_change, correlation], 'P_4':['mean', 'median', 'sum', data_range, iqr, avg_pct_change, correlation]}).reset_index(drop = False)

## Renaming variable names
payment_vars_train.columns = ['customer_ID', 'P_2_mean', 'P_2_median', 'P_2_sum', 'P_2_data_range', 'P_2_iqr', 'P_2_avg_pct_change', 'P_2_correlation', 'P_3_mean', 'P_3_median', 'P_3_sum', 'P_3_data_range', 'P_3_iqr', 'P_3_avg_pct_change', 'P_3_correlation', 'P_4_mean', 'P_4_median', 'P_4_sum', 'P_4_data_range', 'P_4_iqr', 'P_4_avg_pct_change', 'P_4_correlation']

## Creating new Spend features with the cleaned train data-frame
spend_vars_train = train_impute.groupby('customer_ID').agg({'S_3':['median', 'sum', data_range, iqr, avg_pct_change, correlation], 'S_5':[data_range, iqr, avg_pct_change, correlation], 'S_6':['mean', 'median', 'sum', 'mad', data_range, iqr, avg_pct_change, correlation], 'S_8':['mean', 'median', 'sum', data_range, iqr, avg_pct_change, correlation], 'S_13':['mean', 'sum', 'std', 'mad', data_range, iqr, avg_pct_change, correlation], 'S_25':['mean', 'sum', 'std', 'mad', data_range, iqr, avg_pct_change, correlation], 'S_27':[data_range, iqr, avg_pct_change, correlation]}).reset_index(drop = False)

spend_vars_train.columns = ['customer_ID', 'S_3_median', 'S_3_sum', 'S_3_data_range', 'S_3_iqr', 'S_3_avg_pct_change', 'S_3_correlation', 'S_5_data_range', 'S_5_iqr', 'S_5_avg_pct_change', 'S_5_correlation', 'S_6_mean', 'S_6_median', 'S_6_sum', 'S_6_mad', 'S_6_data_range', 'S_6_iqr', 'S_6_avg_pct_change', 'S_6_correlation', 'S_8_mean', 'S_8_median', 'S_8_sum', 'S_8_data_range', 'S_8_iqr', 'S_8_avg_pct_change', 'S_8_correlation', 'S_13_mean', 'S_13_sum', 'S_13_std', 'S_13_mad', 'S_13_data_range', 'S_13_iqr', 'S_13_avg_pct_change', 'S_13_correlation', 'S_25_mean', 'S_25_sum', 'S_25_std', 'S_25_mad', 'S_25_data_range', 'S_25_iqr', 'S_25_avg_pct_change', 'S_25_correlation', 'S_27_data_range', 'S_27_iqr', 'S_27_avg_pct_change', 'S_27_correlation']

## Sanity check
print('-- Training aggregations data-frame complete -- \n')

## -------------------------------------------

## Combining desired count features with other aggregated features to create final data-frame

## Joining the Payment and Spend train data-frames
training = payment_vars_train.merge(spend_vars_train, how = 'left', on = 'customer_ID')

## Joining the Training and Count data-frames
training = training.merge(count_vars_train, how = 'left', on = 'customer_ID')

## Appending target labels to training data-frame
training = training.merge(train_labels, on = 'customer_ID', how = 'left')

## -------------------------------------------

## Exporting the resulting training data-frame to a csv file
training.to_csv('amex_train_payment_spend.csv', index = False)

## Sanity check
print('-- Training data-frame complete -- \n')