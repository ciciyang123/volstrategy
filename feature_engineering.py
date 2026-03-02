# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# Load reference dates
import os
import pickle
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import FeatureHasher



import warnings
warnings.filterwarnings('ignore')

data_folder = 'data'
with open(os.path.join(data_folder, 'refdates.pkl'), 'rb') as f:
    refdatesall = pickle.load(f)

#if a ticker is not in the df_train, it should not be in the df_test
symbols = [symbol for symbol, dates in refdatesall.items() if len(dates) > 1]


# Load the DataFrames
df_train = pd.read_pickle('df_train.pkl')
df_test = pd.read_pickle('df_test.pkl')

# Reset the index to turn 'Date' back into a column
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)

# Now set 'Symbol' as the first index and 'Date' as the second index
df_train.set_index(['Symbol', 'Date'], inplace=True)
df_test.set_index(['Symbol', 'Date'], inplace=True)
#print(df_train, df_test)

symbols = df_test.index.get_level_values('Symbol').unique()

df_test = df_test[df_test.index.get_level_values('Symbol').isin(symbols)]
df_train['return'] = pd.to_numeric(df_train['return'], errors='coerce')
df_train_before_std = df_train.copy()
#df_train['return_pnl'] = df_train['return']
y_train = df_train['return']
#y_train = y_train + np.abs(y_train.min()) + 1
# Apply Box-Cox transformation
#y_train, lambda_ = stats.boxcox(y_train)
#y_train = np.log(y_train)
#print(f'lambda is {lambda_}')
df_train['return'] = y_train

df_train['prertn1'] = df_train.groupby(level = 'Symbol')['return'].shift(1)
df_train['prertn2'] = df_train.groupby(level = 'Symbol')['return'].shift(2)

# Step 2: Drop rows with missing lag-1 or lag-2 values in df_train
df_train = df_train.dropna(subset=['prertn1', 'prertn2'])

# Step 3: Extract the last available lag-1 and lag-2 return values for each symbol from df_train
last_rtns = df_train.groupby('Symbol')[['return', 'prertn1']].last().reset_index()
# Step 4: Merge the lag-1 and lag-2 features with df_test
df_test_reset = df_test.reset_index()
df_test = pd.merge(df_test_reset, last_rtns, on='Symbol', how='left')
df_test = df_test.rename(columns={'prertn1': 'temp_prertn1'})
df_test = df_test.rename(columns = {'return' : 'prertn1', 'temp_prertn1' : 'prertn2'})
# Restore the index after merging
df_test.set_index(['Symbol', 'Date'], inplace=True)
# Drop 'Symbol' from columns if it was added during the merge
if 'Symbol' in df_train.columns:
    df_train = df_train.drop(columns=['Symbol'])
if 'Symbol' in df_test.columns:
    df_test = df_test.drop(columns=['Symbol'])

#encoding


def one_hot_encode_column(df_train, df_test, col_to_encode):
    # Preserve the original index
    original_index_train = df_train.index
    original_index_test = df_test.index
    
    # Perform one-hot encoding on df_train with custom prefixes
    df_train_encoded = pd.get_dummies(df_train, columns=[col_to_encode], prefix=f'Cluster_{col_to_encode}')
    
    # Perform one-hot encoding on df_test with the same custom prefixes
    df_test_encoded = pd.get_dummies(df_test, columns=[col_to_encode], prefix=f'Cluster_{col_to_encode}')
    
    # Ensure the 'return' column is excluded from df_test during alignment
    common_cols = [col for col in df_train_encoded.columns if col in df_test_encoded.columns and col != 'return']
    
    # Align the columns of df_train and df_test based on the common columns
    df_train_encoded = df_train_encoded[common_cols + ['return']]  # Keep 'return' in df_train
    df_test_encoded = df_test_encoded[common_cols]  # Exclude 'return' from df_test
    
    # Re-add the original column to the encoded DataFrames
    df_train_encoded[col_to_encode] = df_train[col_to_encode].values
    df_test_encoded[col_to_encode] = df_test[col_to_encode].values
    
    # Set the original index back
    df_train_encoded.set_index(original_index_train, inplace=True)
    df_test_encoded.set_index(original_index_test, inplace=True)
    
    
    
    return df_train_encoded, df_test_encoded

df_train, df_test = one_hot_encode_column(df_train, df_test, col_to_encode='Industry_Signals')

#df_train, df_test= binary_encode_and_align(df_train, df_test, ['Symbol', 'Industry_Signals'])
    
#print(df_train['Industry_Signals'].nunique(), df_train.shape, df_train.columns)


signal_cols = [col for col in df_train.columns if col!='return']

# Filter only numeric signals
numeric_signals = []
for signal in signal_cols:
    if pd.api.types.is_numeric_dtype(df_train[signal]):
        if signal != 'TurnoverRatio_6m':
            numeric_signals.append(signal)

signal_cols_rtn = numeric_signals + ['return']

#for stv, we need to remove the symbol only has one dp
#symbol_counts = df_train['Symbol'].value_counts()
#symbols_to_drop = symbol_counts[symbol_counts == 1].index
#df_train = df_train[~df_train['Symbol'].isin(symbols_to_drop)]

#df_train = df_train.drop(columns=['Symbol'])
#df_test = df_test.drop(columns=['Symbol'])
# Get the counts of each Symbol using the index level
symbol_counts = df_train.index.get_level_values('Symbol').value_counts()

# Identify symbols that only have one data point
symbols_to_drop = symbol_counts[symbol_counts == 1].index
print(symbols_to_drop)
# Filter out these symbols
df_train = df_train[~df_train.index.get_level_values('Symbol').isin(symbols_to_drop)]
df_test = df_test[~df_test.index.get_level_values('Symbol').isin(symbols_to_drop)]
std_dev = df_train.groupby('Symbol').std()

# Function to normalize the DataFrame
def normalize_train(df_train, std_dev, isX = 1):
    df_normalized = df_train.copy()
    # Exclude the cluster columns from normalization
    cluster_cols = [col for col in df_train.columns if not col.startswith('Cluster')]
   
    # Define the columns to be normalized
    cols_to_normalize = [col for col in signal_cols_rtn if col in cluster_cols] if isX else [col for col in numeric_signals if col in cluster_cols]
 
    #df_normalized.loc[df.index.get_level_values('Symbol') == symbol, columns_to_normalize] /= std_dev.loc[symbol, columns_to_normalize]
    
    for symbol in symbols:
        if symbol in std_dev.index:
            # Use a tuple to correctly reference the multi-index DataFrame
            if isX:
                df_normalized.loc[df_normalized.index.get_level_values('Symbol') == symbol, cols_to_normalize] /= std_dev.loc[symbol, cols_to_normalize]
        
    
    return df_normalized

def normalize_test(df_test, std_dev, isX = 0):
    df_normalized = df_test.copy()
    cluster_cols = [col for col in df_test.columns if not col.startswith('Cluster')]
    cols_to_normalize = [col for col in numeric_signals if col in cluster_cols]
    #df_normalized.loc[:, cols_to_normalize] /= std_dev.loc[:, cols_to_normalize]
    for symbol in df_normalized.index.get_level_values('Symbol').unique():
        if symbol in std_dev.index:
            # Since there's only one row per symbol in df_test, we can directly divide by std_dev
            df_normalized.loc[df_normalized.index.get_level_values('Symbol') == symbol, cols_to_normalize] /= std_dev.loc[symbol, cols_to_normalize]
    
    return df_normalized

# Normalize df_train and df_test
df_train_normalized = normalize_train(df_train, std_dev, 1)
df_test_normalized = normalize_test(df_test, std_dev, 0)
#print(df_train_normalized, df_test_normalized)
# Prepare data for regression
X_train = df_train_normalized[numeric_signals]
y_train = df_train_normalized['return']
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = df_test_normalized[numeric_signals]

#save those df
# Save X_train
with open('X_FEtrain.pkl', 'wb') as file:
    pickle.dump(X_train, file)

# Save y_train
with open('y_FEtrain.pkl', 'wb') as file:
    pickle.dump(y_train, file)

# Save X_test
with open('X_FEtest.pkl', 'wb') as file:
    pickle.dump(X_test, file)

with open('X_train_before_std.pkl', 'wb') as file:
    pickle.dump(df_train_before_std, file)

with open('std_dev_return.pkl', 'wb') as f:
    pickle.dump(std_dev['return'], f)
#Fit the model


### OLS Model
"""
X_train_with_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_with_constant).fit()
print(model.summary())
# Prepare test data for prediction
X_test = sm.add_constant(X_test)
#print(X_train_with_constant.shape)  # This should match the number of coefficients in your model
#print(X_test.shape)  # This should match the above

# Make predictions
predictions = model.predict(X_test)
# Transform the predicted returns back to the original scale


df_test_normalized['return'] = predictions
for symbol in symbols:
    if symbol in std_dev.index:
        df_test_normalized.loc[df_test.index.get_level_values('Symbol') == symbol, 'return'] *= std_dev.loc[symbol, 'return']
"""