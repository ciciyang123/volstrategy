# -*- coding: utf-8 -*-
"""
Integrate FE and RollingModel
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from scipy import stats

class FeatureEngineering:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.df_train = pd.read_pickle(os.path.join(data_folder, 'df_train.pkl'))
        self.df_test = pd.read_pickle(os.path.join(data_folder, 'df_test.pkl'))
        self.df_train_before_std = self.df_train.copy()

    def prepare_data(self):
        # Load reference dates and symbols
        with open(os.path.join(self.data_folder, 'data','refdates.pkl'), 'rb') as f:
            refdatesall = pickle.load(f)

        symbols = [symbol for symbol, dates in refdatesall.items() if len(dates) > 1]
        
        # Filter df_train and df_test based on the symbols
        self.df_test = self.df_test[self.df_test.index.get_level_values('Symbol').isin(symbols)]
        
        self.df_train['return'] = pd.to_numeric(self.df_train['return'], errors='coerce')
        
        # Feature engineering: lagged returns
        self.df_train['prertn1'] = self.df_train.groupby(level='Symbol')['return'].shift(1)
        self.df_train['prertn2'] = self.df_train.groupby(level='Symbol')['return'].shift(2)

        # Drop rows with missing lag-1 or lag-2 values in df_train
        self.df_train = self.df_train.dropna(subset=['prertn1', 'prertn2'])
        
        # Extract the last available lag-1 and lag-2 return values for each symbol from df_train
        last_rtns = self.df_train.groupby('Symbol')[['return', 'prertn1']].last().reset_index()

        # Merge the lag-1 and lag-2 features with df_test
        df_test_reset = self.df_test.reset_index()
        self.df_test = pd.merge(df_test_reset, last_rtns, on='Symbol', how='left')
        self.df_test = self.df_test.rename(columns={'prertn1': 'temp_prertn1'})
        self.df_test = self.df_test.rename(columns={'return': 'prertn1', 'temp_prertn1': 'prertn2'})

        # Restore the index after merging
        self.df_test.set_index(['Symbol', 'Date'], inplace=True)
        
        # Drop 'Symbol' from columns if it was added during the merge
        if 'Symbol' in self.df_train.columns:
            self.df_train = self.df_train.drop(columns=['Symbol'])
        if 'Symbol' in self.df_test.columns:
            self.df_test = self.df_test.drop(columns=['Symbol'])
        
        
        # One-hot encode the 'Industry_Signals' column
        self.df_train, self.df_test = self._one_hot_encode_column(self.df_train, self.df_test, 'Industry_Signals')

       
        
        # Check for NaNs after one-hot encoding and filtering
        #print("NaNs after one-hot encoding and filtering:")
        #print(self.df_train.isnull().sum())
        #print(self.df_test.isnull().sum())
        # Save intermediate results
        with open(os.path.join(self.data_folder, 'X_train_before_std.pkl'), 'wb') as file:
            pickle.dump(self.df_train_before_std, file)
        
    def run_backtest(self, refdate):
        return self._run(refdate, goal = "backtest")
    
    def run_predict(self, refdate):
        return self._run(refdate, goal = "predict")
    
    def _run(self, refdate, goal):
        if goal == "backtest":
            self.df_train_split = self.df_train[self.df_train.index.get_level_values('Date') < refdate]
            self.df_test_split = self.df_train[self.df_train.index.get_level_values('Date') == refdate]
            #print(self.df_train_split.loc[self.df_train_split.index.get_level_values("Symbol") == "VLTO"])
            # Remove symbols with only one data point
            symbol_counts = self.df_train_split.index.get_level_values('Symbol').value_counts()
            symbols_to_drop = symbol_counts[symbol_counts == 1].index
            self.df_train_split = self.df_train_split[~self.df_train_split.index.get_level_values('Symbol').isin(symbols_to_drop)]
            self.df_test_split = self.df_test_split[~self.df_test_split.index.get_level_values('Symbol').isin(symbols_to_drop)]
            std_dev = self.df_train_split.groupby('Symbol').std()
            ###
            
            
            df_train_normalized, numeric_signals = self._normalize(self.df_train_split, std_dev)
            
            
            df_test_normalized_split, numeric_signals = self._normalize(self.df_test_split, std_dev, is_train=False)
            #df_test_normalized, numeric_signals = self._normalize(self.df_test, std_dev, is_train=False)
            X_train_split = df_train_normalized[numeric_signals]
            y_train_split = df_train_normalized['return']
            X_train_split = X_train_split.apply(pd.to_numeric, errors='coerce')
            X_valid = df_test_normalized_split[numeric_signals]
            #X_test = df_test_normalized[numeric_signals]
            return X_train_split, y_train_split, X_valid, std_dev
        
        elif goal == "predict":
            self.df_train = self.df_train[self.df_train.index.get_level_values('Date') < refdate]
             # Remove symbols with only one data point
            symbol_counts = self.df_train.index.get_level_values('Symbol').value_counts()
            symbols_to_drop = symbol_counts[symbol_counts == 1].index
            self.df_train = self.df_train[~self.df_train.index.get_level_values('Symbol').isin(symbols_to_drop)]
            self.df_test = self.df_test[~self.df_test.index.get_level_values('Symbol').isin(symbols_to_drop)]
            std_dev = self.df_train.groupby('Symbol').std()
            df_train_normalized, numeric_signals = self._normalize(self.df_train, std_dev)
            df_test_normalized, numeric_signals = self._normalize(self.df_test, std_dev, is_train=False)
            X_train_split = df_train_normalized[numeric_signals]
            y_train_split = df_train_normalized['return']
            X_train_split = X_train_split.apply(pd.to_numeric, errors='coerce')
            X_test = df_test_normalized[numeric_signals]
            return X_train_split, y_train_split, X_test, std_dev
            
            

    def _normalize(self, df, std_dev, is_train=True):
        df_normalized = df.copy()
        cluster_cols = [col for col in df.columns if not col.startswith('Cluster')]
        signal_cols = [col for col in df.columns if col!='return']

        # Filter only numeric signals
        numeric_signals = []
        for signal in signal_cols:
            if pd.api.types.is_numeric_dtype(df[signal]):
                if signal != 'TurnoverRatio_6m':
                    numeric_signals.append(signal)
        
        signal_cols_rtn = numeric_signals + ['return']
        
        cols_to_normalize = [col for col in signal_cols_rtn if col in cluster_cols] if is_train else [col for col in numeric_signals if col in cluster_cols]
        for symbol in df.index.get_level_values('Symbol').unique():
            if symbol in std_dev.index:
                df_normalized.loc[df_normalized.index.get_level_values('Symbol') == symbol, cols_to_normalize] /= std_dev.loc[symbol, cols_to_normalize]
        df_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_normalized.dropna(inplace = True)
        return df_normalized, numeric_signals
    
    def _one_hot_encode_column(self, df_train, df_test, col_to_encode):
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



class RollingBacktest:
    def __init__(self, n_symbols, percentile, model_class, param_grid, data_folder):
        self.model_class = model_class  # Ridge or Lasso
        self.param_grid = param_grid
        self.data_folder = data_folder
        self.models = {}  # To store fitted models
        self.alpha_parameters = {}  # To store alpha parameters
        self.percentile = percentile
        self.n = n_symbols
        self.liquidity_ratio_train = pd.read_pickle(os.path.join(data_folder, 'liquidity_train.pkl'))
        self.liquidity_ratio_test = pd.read_pickle(os.path.join(data_folder, 'liquidity_test.pkl'))
        self.marketcap_train = pd.read_pickle(os.path.join(data_folder, 'marketcap_train.pkl'))
        self.marketcap_test = pd.read_pickle(os.path.join(data_folder, 'marketcap_test.pkl'))
        self.feature_engineering = FeatureEngineering(data_folder)  # Initialize FeatureEngineering
        self.feature_engineering.prepare_data()  # Prepare data before any backtesting or prediction
        with open(os.path.join(self.data_folder, 'X_train_before_std.pkl'), 'rb') as file:
            self.df_train_before_std = pickle.load(file)
    
    def run_backtest(self, past_i, strategytype="ShortOnly", goal="backtest"):
        validation_date = self._get_validation_date(past_i, goal)
        if goal == "backtest":
            X_train_split, y_train_split, X_valid, std_dev = self._prepare_data_backtest(validation_date)
            if past_i == 11:
                print(np.isinf(X_train_split).sum(), np.isinf(y_train_split).sum())
            self._fit_model(X_train_split, y_train_split, validation_date)
            #self.calculate_baseline(validation_date)
            return self._test_model(X_valid, std_dev, validation_date, strategytype)
        elif goal == "predict":
            X_train_split, y_train_split, X_test, std_dev = self._prepare_data_predict(validation_date)
            self._fit_model(X_train_split, y_train_split, validation_date)
            return self._predict_y(X_test, std_dev, validation_date)
    
    def _get_validation_date(self, past_i, goal):
        if goal == "backtest":
            unique_dates = self.df_train_before_std.index.get_level_values('Date').unique()
            sorted_dates = sorted(unique_dates, reverse=True)
            # Ensure past_i is within bounds
            if past_i < 0 or past_i >= len(sorted_dates):
                raise IndexError(f"past_i {past_i} is out of bounds for available dates.")
        
            
            return sorted_dates[past_i]
            
        elif goal == "predict":
            return self.feature_engineering.df_test.index.get_level_values('Date').unique()[0]
             
            

    def _prepare_data_backtest(self, validation_date):
        return self.feature_engineering.run_backtest(validation_date)
    
    def _prepare_data_predict(self, validation_date):
        return self.feature_engineering.run_predict(validation_date)
    
    def _fit_model(self, X_train, y_train, validation_date):
        
        
        # Align the liquidity weights with the training data
        weights = 0.8 * self.liquidity_ratio_train.loc[X_train.index] + 0.2 * self.marketcap_train.loc[X_train.index]
        
        model_cv = GridSearchCV(self.model_class(), self.param_grid, cv=5)
        model_cv.fit(X_train, y_train, sample_weight= weights)
        
        best_alpha = model_cv.best_params_['alpha']
        self.alpha_parameters[validation_date] = best_alpha
        
        model = self.model_class(alpha=best_alpha)
        model.fit(X_train, y_train, sample_weight=weights)
        
        self.models[validation_date] = model

    def _test_model(self, X_valid, std_dev, validation_date, strategytype):
        model = self.models[validation_date]
        y_pred = model.predict(X_valid)
        
        top_symbols = self._revert_predictions(y_pred, X_valid, std_dev, validation_date)
        return self._calculate_pnl(top_symbols, validation_date, strategytype)
    
    def _predict_y(self, X_test, std_dev, validation_date):
        model = self.models[validation_date]
        y_pred = model.predict(X_test)
        Short_top_symbols = self._revert_predictions(y_pred, X_test, std_dev, validation_date, goal = "predict")[:self.n]
        Long_top_symbols = self._revert_predictions(y_pred, X_test, std_dev, validation_date, goal = "predict")[-self.n:]
        ss = [symbol for symbol, _ in Short_top_symbols]
        ls = [symbol for symbol, _ in Long_top_symbols]
        return ss, ls

    def _revert_predictions(self, predictions, X_valid, std_dev, refdate, goal = "backtest"):
        reverted_returns = []
        res = []
        for idx, symbol in enumerate(X_valid.index.get_level_values('Symbol')):
            if symbol in std_dev.index:
                symbol_std = std_dev.loc[symbol, 'return']
                pred_return = predictions[idx] * symbol_std
                reverted_returns.append((symbol, pred_return))
        if goal == "backtest":
            weights = self.liquidity_ratio_train.loc[X_valid.index]
        elif goal == "predict":
            weights = self.liquidity_ratio_test.loc[X_valid.index]
        threshold = weights.quantile(self.percentile)
        for symbol, pr in reverted_returns:
            if symbol != "SRCL": #exclude some symbols
                sw = weights.loc[(symbol, refdate)]
                if sw > threshold:
                    res.append((symbol, pr))
                
            
        return sorted(res, key=lambda x: x[1], reverse=True)

    def _calculate_pnl(self, top_symbols, refdate, strategytype):
        symbols = top_symbols[:self.n]
        if strategytype == "ShortOnly":
            pnl = sum(self.df_train_before_std.loc[(symbol, refdate), 'return'] for symbol, _ in symbols)
        elif strategytype == "ShortLong":
            Lsymbols = top_symbols[-self.n:]
            #print(f"On {refdate}, Short Symbols are {[symbol for symbol, _ in symbols]}")
            #print(f"On {refdate}, Long Symbols are {[symbol for symbol, _ in Lsymbols]}")
            #for symbol, _ in symbols:
            #    print(f"On {refdate} for {symbol}, Short return is {self.df_train_before_std.loc[(symbol, refdate), 'return']}")
            #for symbol, _ in Lsymbols:
            #    print(f"On {refdate} for {symbol}, Long return is {-self.df_train_before_std.loc[(symbol, refdate), 'return']}")
             
            Spnl = sum(self.df_train_before_std.loc[(symbol, refdate), 'return'] for symbol, _ in symbols)
            Lpnl = sum(-self.df_train_before_std.loc[(symbol, refdate), 'return'] for symbol, _ in Lsymbols)
            pnl = Spnl + Lpnl
        print(f'PnL for {refdate}: {pnl}')
        return pnl
    
    def calculate_baseline(self, validation_date):
        # Filter df for the validation date
        df_validation = self.df_train_before_std.loc[self.df_train_before_std.index.get_level_values('Date') == validation_date]

        # Filter positive and negative returns within the validation date
        positive_returns = df_validation[df_validation['return'] > 0]
        negative_returns = df_validation[df_validation['return'] < 0]

        # Calculate the number, mean, and standard deviation for positive returns
        num_positive = positive_returns['return'].count()
        mean_positive = positive_returns['return'].mean()
        std_positive = positive_returns['return'].std()

        # Calculate the number, mean, and standard deviation for negative returns
        num_negative = negative_returns['return'].count()
        mean_negative = negative_returns['return'].mean()
        std_negative = negative_returns['return'].std()

        # Print the results
        print(f"{validation_date} : Positive Returns: {num_positive}, Mean: {mean_positive}, Std Dev: {std_positive}")
        print(f"{validation_date} : Negative Returns: {num_negative}, Mean: {mean_negative}, Std Dev: {std_negative}")

data_folder = 'C:/GSaTasks/Others/VolStrategy1'
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
backtester = RollingBacktest(50, 0.7, Ridge, param_grid, data_folder)


ss, ls = backtester.run_backtest(0, strategytype = "ShortLong", goal = "predict")
# Running prediction: change goal from backtest to predict
#ss, ls = backtester.run_backtest(2024, 8, strategytype="ShortOnly", goal = "predict")
print(f"Short Symbols : {ss}")
print(f"Long Symbols : {ls}")


def calculate_sharpe_ratio(pnl_results):
        pnl_array = np.array(pnl_results)
        mean_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array)
        sharpe_ratio = mean_pnl / std_pnl
        return sharpe_ratio

"""
pnl_lst = []
for i in range(11):
   pnl_lst.append(backtester.run_backtest(i, strategytype = "ShortLong", goal = "backtest"))
   

print(calculate_sharpe_ratio(pnl_lst))
"""

