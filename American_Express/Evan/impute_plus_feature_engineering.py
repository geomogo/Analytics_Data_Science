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

## Appending target labels with train data-frame
train = train.merge(train_labels, on = 'customer_ID', how = 'left')

print(train.columns)

## Sanity check
print('-- Data Read -- \n')

## -------------------------------------------

############################################
## Payment and Spend variable engineering ##
############################################

# ## Computing new Payment and Spend count features with the train data-frame
# count_vars_train = train.groupby('customer_ID').agg({'P_3':['count'], 'S_3':['count'], 'S_7':['count'], 
#                                                      'S_22': ['count'], 'S_23': ['count'], 'S_25': ['count']}).reset_index(drop = False)

# ## Editing the variable names in the data-frame
# count_vars_train.columns = ['customer_ID', 'P_3_count', 'S_3_count', 'S_7_count', 'S_22_count', 'S_23_count', 'S_25_count']

##################################
## Balance variable engineering ##
##################################

## Computing new Balance count features with the train data-frame
count_vars_train = train.groupby('customer_ID').agg({'B_2':['count'], 'B_3':['count'], 'B_6':['count'], 'B_8': ['count'], 'B_13': ['count'], 'B_15': ['count'], 'B_16': ['count'], 'B_17': ['count'], 'B_19': ['count'], 'B_20': ['count'], 'B_22': ['count'], 'B_25': ['count'], 'B_26': ['count'], 'B_27': ['count'], 'B_29': ['count'], 'B_30': ['count'], 'B_33': ['count'], 'B_37': ['count'], 'B_38': ['count'], 'B_39': ['count'], 'B_40': ['count'], 'B_41': ['count'], 'B_42': ['count']}).reset_index(drop = False)

## Editing the variable names in the data-frame
count_vars_train.columns = ['customer_ID', 'B_2_count', 'B_3_count', 'B_6_count', 'B_8_count', 'B_13_count', 'B_15_count', 'B_16_count', 'B_17_count', 'B_19_count', 'B_20_count', 'B_22_count', 'B_25_count', 'B_26_count', 'B_27_count', 'B_29_count', 'B_30_count', 'B_33_count', 'B_37_count','B_38_count', 'B_39_count', 'B_40_count', 'B_41_count', 'B_42_count']

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
def correlation(x):
    return pd.Series(x.values).corr(other = pd.Series(x.index), method = 'pearson')

############################################
## Payment and Spend variable engineering ##
############################################

# ## Creating new Payment features with the cleaned train data-frame
# payment_vars_train = train_impute.groupby('customer_ID').agg({'P_2':['mean', 'median', 'sum', data_range, iqr, correlation], 'P_3':['mean', 'median', 'sum', data_range, iqr, correlation], 'P_4':['mean', 'median', 'sum', data_range, iqr, correlation]}).reset_index(drop = False)

# ## Renaming variable names
# payment_vars_train.columns = ['customer_ID', 'P_2_mean', 'P_2_median', 'P_2_sum', 'P_2_data_range', 'P_2_iqr', 'P_2_correlation', 'P_3_mean', 'P_3_median', 'P_3_sum', 'P_3_data_range', 'P_3_iqr', 'P_3_correlation', 'P_4_mean', 'P_4_median', 'P_4_sum', 'P_4_data_range', 'P_4_iqr', 'P_4_correlation']

# ## Creating new Spend features with the cleaned train data-frame
# spend_vars_train = train_impute.groupby('customer_ID').agg({'S_3':['median', 'sum', data_range, iqr, correlation], 'S_5':[data_range, iqr, correlation], 'S_6':['mean', 'median', 'sum', 'mad', data_range, iqr, correlation], 'S_8':['mean', 'median', 'sum', data_range, iqr, correlation], 'S_13':['mean', 'sum', 'std', 'mad', data_range, iqr, correlation], 'S_25':['mean', 'sum', 'std', 'mad', data_range, iqr, correlation], 'S_27':[data_range, iqr, correlation]}).reset_index(drop = False)

# spend_vars_train.columns = ['customer_ID', 'S_3_median', 'S_3_sum', 'S_3_data_range', 'S_3_iqr', 'S_3_correlation', 'S_5_data_range', 'S_5_iqr', 'S_5_correlation', 'S_6_mean', 'S_6_median', 'S_6_sum', 'S_6_mad', 'S_6_data_range', 'S_6_iqr', 'S_6_correlation', 'S_8_mean', 'S_8_median', 'S_8_sum', 'S_8_data_range', 'S_8_iqr', 'S_8_correlation', 'S_13_mean', 'S_13_sum', 'S_13_std', 'S_13_mad', 'S_13_data_range', 'S_13_iqr', 'S_13_correlation', 'S_25_mean', 'S_25_sum', 'S_25_std', 'S_25_mad', 'S_25_data_range', 'S_25_iqr', 'S_25_correlation', 'S_27_data_range', 'S_27_iqr', 'S_27_correlation']

##################################
## Balance variable engineering ##
##################################

balance_vars_train = train_impute.groupby('customer_ID').agg({'B_1':['mean', 'median', 'std', data_range, iqr, correlation], 'B_2':['std', data_range, iqr, correlation], 'B_3':['mean', 'median', 'std', data_range, iqr, correlation], 'B_4':['mean', 'median', 'std', data_range, iqr, correlation], 'B_5':['mean', 'median', 'std', data_range, iqr, correlation], 'B_6':['median', data_range, iqr, correlation], 'B_7':['mean', 'median', 'std', data_range, iqr, correlation], 'B_9':['mean', 'median', 'std', data_range, iqr, correlation], 'B_11':['mean', 'median', 'std', data_range, iqr, correlation], 'B_12':['mean', 'median', 'std', data_range, iqr, correlation], 'B_13': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_14':['mean', 'median', 'std', data_range, iqr, correlation], 'B_15': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_16': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_17': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_18':['mean', 'median', 'std', data_range, iqr, correlation], 'B_19': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_20': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_23':['mean', 'median', 'std', data_range, iqr, correlation], 'B_30': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_33':['mean', 'median', 'std', data_range, iqr, correlation], 'B_38': ['mean', 'median', 'std', data_range, iqr, correlation], 'B_42': ['mean', 'median', 'std', data_range, iqr, correlation]}).reset_index(drop = False)

balance_vars_train.columns = ['customer_ID', 'B_1_mean', 'B_1_median', 'B_1_std', 'B_1_data_range', 'B_1_iqr', 'B_1_correlation', 'B_2_std', 'B_2_data_range', 'B_2_iqr', 'B_2_correlation', 'B_3_mean', 'B_3_median', 'B_3_std', 'B_3_data_range', 'B_3_iqr', 'B_3_correlation', 'B_4_mean', 'B_4_median', 'B_4_std', 'B_4_data_range', 'B_4_iqr', 'B_4_correlation', 'B_5_mean', 'B_5_median', 'B_5_std', 'B_5_data_range', 'B_5_iqr', 'B_5_correlation', 'B_6_median', 'B_6_data_range', 'B_6_iqr', 'B_6_correlation', 'B_7_mean', 'B_7_median', 'B_7_std', 'B_7_data_range', 'B_7_iqr', 'B_7_correlation', 'B_9_mean', 'B_9_median', 'B_9_std', 'B_9_data_range', 'B_9_iqr', 'B_9_correlation', 'B_11_mean', 'B_11_median', 'B_1_std', 'B_11_data_range', 'B_11_iqr', 'B_11_correlation', 'B_12_mean', 'B_12_median', 'B_12_std', 'B_12_data_range', 'B_12_iqr', 'B_12_correlation', 'B_13_mean', 'B_13_median', 'B_13_std', 'B_13_data_range', 'B_13_iqr', 'B_13_correlation', 'B_14_mean', 'B_14_median', 'B_14_std', 'B_14_data_range', 'B_14_iqr', 'B_14_correlation', 'B_15_mean', 'B_15_median', 'B_15_std', 'B_15_data_range', 'B_15_iqr', 'B_15_correlation', 'B_16_mean', 'B_16_median', 'B_16_std', 'B_16_data_range', 'B_16_iqr', 'B_16_correlation', 'B_17_mean', 'B_17_median', 'B_17_std', 'B_17_data_range', 'B_17_iqr', 'B_17_correlation', 'B_18_mean', 'B_18_median', 'B_18_std', 'B_18_data_range', 'B_18_iqr', 'B_18_correlation', 'B_19_mean', 'B_19_median', 'B_19_std', 'B_19_data_range', 'B_19_iqr', 'B_19_correlation', 'B_20_mean', 'B_20_median', 'B_20_std', 'B_20_data_range', 'B_20_iqr', 'B_20_correlation', 'B_23_mean', 'B_23_median', 'B_23_std', 'B_23_data_range', 'B_23_iqr', 'B_23_correlation', 'B_30_mean', 'B_30_median', 'B_30_std', 'B_30_data_range', 'B_30_iqr', 'B_30_correlation', 'B_33_mean', 'B_33_median', 'B_33_std', 'B_33_data_range', 'B_33_iqr', 'B_33_correlation', 'B_38_mean', 'B_38_median', 'B_38_std', 'B_38_data_range', 'B_38_iqr', 'B_38_correlation', 'B_42_mean', 'B_42_median', 'B_42_std', 'B_42_data_range', 'B_42_iqr', 'B_42_correlation']

## Sanity check
print('-- Training aggregations data-frame complete -- \n')

## -------------------------------------------

## Combining desired count features with other aggregated features to create final data-frame

# ## Joining the Payment and Spend train data-frames
# training = payment_vars_train.merge(spend_vars_train, how = 'left', on = 'customer_ID')

# ## Joining the Training and Count data-frames
# training = training.merge(count_vars_train, how = 'left', on = 'customer_ID')

# ## Appending target labels to training data-frame
# training = training.merge(train_labels, on = 'customer_ID', how = 'left')

## Joining the Training and Count data-frames
training = balance_vars_train.merge(count_vars_train, how = 'left', on = 'customer_ID')

## Appending target labels to training data-frame
training = training.merge(train_labels, on = 'customer_ID', how = 'left')

## -------------------------------------------

# ## Exporting the resulting training data-frame to a csv file
# training.to_csv('amex_train_payment_spend.csv', index = False)

## Exporting the resulting training data-frame to a csv file
training.to_csv('amex_train_balance.csv', index = False)

## Sanity check
print('-- Training data-frame complete -- \n')