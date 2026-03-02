# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import pickle
from pipeline import Pipeline
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

# Load reference dates
data_folder = 'data'
with open(os.path.join(data_folder, 'refdates.pkl'), 'rb') as f:
    refdatesall = pickle.load(f)


# Run the pipeline to get the training and testing data
df_train, df_test = Pipeline.run(data_folder, refdatesall)

###some RealIv are 0, then drop
df_train.dropna(inplace = True)
# Reset the index to turn 'Date' back into a column
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)

# Now set 'Symbol' as the first index and 'Date' as the second index
df_train.set_index(['Symbol', 'Date'], inplace=True)
df_test.set_index(['Symbol', 'Date'], inplace=True)



# Assuming liquidity_ratio is a column in df_train and df_test
df_train['Liquidity_Ratio_Standardized'] = df_train['Liquidity_Ratio'].apply(lambda x: x / df_train['Liquidity_Ratio'].std())
df_test['Liquidity_Ratio_Standardized'] = df_test['Liquidity_Ratio'].apply(lambda x: x / df_test['Liquidity_Ratio'].std())
df_train['MarketCap_Standardized'] = df_train['MarketCap'].apply(lambda x: x / df_train['MarketCap'].std())
df_test['MarketCap_Standardized'] = df_test['MarketCap'].apply(lambda x: x / df_test['MarketCap'].std())
# Set MultiIndex and align with X_train and X_test
liquidity_ratio_train = df_train['Liquidity_Ratio_Standardized']
liquidity_ratio_test = df_test['Liquidity_Ratio_Standardized']
marketcap_train = df_train['MarketCap_Standardized']
marketcap_test = df_test['MarketCap_Standardized']

# Save the standardized liquidity ratio to separate files
with open('liquidity_train.pkl', 'wb') as file:
    pickle.dump(liquidity_ratio_train, file)

with open('liquidity_test.pkl', 'wb') as file:
    pickle.dump(liquidity_ratio_test, file)
    
with open('marketcap_train.pkl', 'wb') as file:
    pickle.dump(marketcap_train, file)
with open('marketcap_test.pkl', 'wb') as file:
    pickle.dump(marketcap_test, file)
# Drop the liquidity_ratio column from df_train and df_test
df_train = df_train.drop(columns=['Liquidity_Ratio', 'Liquidity_Ratio_Standardized', 'MarketCap', 'MarketCap_Standardized'])
df_test = df_test.drop(columns=['Liquidity_Ratio', 'Liquidity_Ratio_Standardized', 'MarketCap', 'MarketCap_Standardized'])

# Save files to pickle files
df_train.to_pickle('df_train.pkl')
df_test.to_pickle('df_test.pkl')


print(df_train, df_test)
print(df_train.isnull().sum())
print(df_test.isnull().sum())
print(np.isinf(df_train).sum())  # Sum of infinite values per column
print(np.isinf(df_train).sum())
print(liquidity_ratio_train, liquidity_ratio_test)
