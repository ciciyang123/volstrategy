# -*- coding: utf-8 -*-

# volatility_signal.py
from signal_base import BaseSignal
import numpy as np
import pandas as pd
import os
from pandas.tseries.offsets import BDay, DateOffset
from sklearn.linear_model import LinearRegression

cd = os.getcwd()
vixf = 'vix_price_return.pkl'
file_path = os.path.join(cd, 'data', vixf)
#volatilities = df['cencc180']
#iv_series = df['cIV180']
class EWMA_Signal(BaseSignal):
    def __init__(self, symbol, data, refdates, lambda_=0.94):
        super().__init__(symbol, data, refdates)
        self.lambda_ = lambda_
        self.volatilities = data['cencc180']
        self.iv_series = data['cIV180']
        self.signal_name =  'EWMA_Signal'

    def calculate_ewma_refdate(self, refdate):
        start_date = pd.to_datetime(refdate) - BDay(126)
        end_date = pd.to_datetime(refdate)
        
        volatilities = self.volatilities[(self.volatilities.index >= start_date) & (self.volatilities.index <= end_date)]
        
        
        if len(volatilities) < 126:
            print(len(volatilities))
            raise ValueError("Not enough data to calculate EWMA.")
        
        ewma_squared = np.zeros(len(volatilities))
        ewma_squared[0] = volatilities.iloc[0] ** 2
        for t in range(1, len(volatilities)):
            ewma_squared[t] = self.lambda_ * ewma_squared[t - 1] + (1 - self.lambda_) * volatilities.iloc[t] ** 2
        ewma = np.sqrt(ewma_squared[-1])
        return ewma
    
    def calculate_signal(self, refdate):
        try:
            current_iv = self.iv_series.loc[refdate]
            current_ewma = self.calculate_ewma_refdate(refdate)
            signal = (current_iv - current_ewma) / current_ewma * 100
            return {self.signal_name : signal}
        except ValueError as e:
            raise e

class Quantile_Signal(BaseSignal):
    def __init__(self, symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.iv_series = data['cIV180']
        self.signal_name = 'Quantile_Signal'

    def calculate_signal(self, refdate):
        start_date = pd.to_datetime(refdate) - BDay(126)
        end_date = pd.to_datetime(refdate)
        ivs = self.iv_series[(self.iv_series.index >= start_date) & (self.iv_series.index <= end_date)]
        ivs_pct_change = ivs.pct_change().dropna() * 100
        # Replace infinite values with NaN, and then drop them
        ivs_pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
        ivs_pct_change.dropna(inplace=True)
        vixs = pd.read_pickle(file_path)
        vixs.index = pd.to_datetime(vixs.index)
        #Aligh both
        common_dates = ivs_pct_change.index.intersection(vixs.index)
        #print(f'ivs_pct_change : {ivs_pct_change}, vixs : {vixs}')
        ivs_pct_change_aligned = ivs_pct_change.loc[common_dates]
        vixs_aligned = vixs.loc[common_dates]
        
        a = ivs_pct_change_aligned.isna().sum().sum()
        b = vixs_aligned.isna().sum().sum()
        
        c = np.isinf(ivs_pct_change_aligned).sum().sum()
        d = np.isinf(vixs_aligned).sum().sum()
        if a > 0:
            print(f'{self.symbol} on {refdate}: {a} NaNs in ivs_pct_change_aligned')
        if b > 0:
            print(f'{self.symbol} on {refdate}: {b} NaNs in vixs_aligned')
        if c > 0:
            print(f'{self.symbol} on {refdate}: {c} infinite values in ivs_pct_change_aligned')
        if d > 0:
            print(f'{self.symbol} on {refdate}: {d} infinite values in vixs_aligned')
                
            

        # Reshape the data for sklearn (requires 2D array)
        X = vixs_aligned.values.reshape(-1, 1)
        y = ivs_pct_change_aligned.values
        
        # Perform linear regression
        model = LinearRegression().fit(X, y)
        sensitivity = model.coef_[0]
        
        if len(ivs) < 126:
            raise ValueError("Not enough data to calculate quantile.")
        
        current_iv = self.iv_series.loc[refdate]
        quantile = (ivs <= current_iv).sum() / len(ivs)
        return {self.signal_name : quantile, 'IVBeta' : sensitivity}

class RSI_Signal(BaseSignal):
    def __init__(self, symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.rsi_series = self.data['rsi']
        self.signal_name = 'RSI_Signal'
    
    def calculate_signal(self, refdate):
        rsi_value = self.rsi_series.loc[refdate]
        return {self.signal_name: rsi_value}
    
class TurnoverRatio_Signal(BaseSignal):
    def __init__(self, symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.signal_name = 'TurnoverRatio_Signals'

    def calculate_signal(self, refdate):
        turnover_ratio = self.data['turnover'].loc[refdate]
        turnover_ratio_3m = self.data['turnover3m'].loc[refdate]
        turnover_ratio_6m = self.data['turnover6m'].loc[refdate]
       
        
        return {
            'TurnoverRatio': turnover_ratio,
            'TurnoverRatio_3m': turnover_ratio_3m,
            'TurnoverRatio_6m': turnover_ratio_6m
        }
    

class Industry_Signal(BaseSignal):
    def __init__(self, symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.signal_name = 'Industry_Signals'
    
    def calculate_signal(self, refdate):
        subgroup = self.data['industrysubgroup'].dropna()
        ugroup = subgroup.unique()
        if len(ugroup) == 1:
            return {self.signal_name : ugroup[0]} 
        else:
            print(f'{self.symbol} do not have industry name')
            return {self.signal_name : None}
        
        

class HVIV_Signal(BaseSignal):
    def __init__(self,symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.signal_name = 'HVIV_Signals'
    
    def calculate_signal(self, refdate):
        iv180 = self.data['impvol180'].loc[refdate]
        hv5 = self.data['cencc5'].loc[refdate]
        civ90 = self.data['cIV90'].loc[refdate]
        rv = self.data['cc30'].loc[refdate]
        hviv = (hv5 - civ90)/ hv5 * 100
        rvivsq = rv**2 / iv180**2 * 100
        return {self.signal_name : hviv, 'iv' : iv180, 'RV_IV_square' : rvivsq}

class HVtrend_Signal(BaseSignal):
    def __init__(self,symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.signal_name = 'HVtrend_Signals'
    
    def calculate_signal(self, refdate):
        hv5 = self.data['cencc5'].loc[refdate]
        hv10 = self.data['cencc10'].loc[refdate]
        hvtrend = (hv5 - hv10)/ hv5 * 100
        return {self.signal_name : hvtrend}

class IVtrend_Signal(BaseSignal):
    def __init__(self, symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.signal_name = 'IVtrend_Signals'
        self.window = 10
    
    def calculate_signal(self, refdate):
        refdate = pd.to_datetime(refdate)
        df = self.data[self.data.index <= refdate]
        ma = df['impvol10'].rolling(window=self.window).mean().iloc[-1]
        current_iv = df.loc[refdate, 'impvol10']
        iv_trend_signal = (current_iv - ma) / ma * 100
        return {self.signal_name: iv_trend_signal}

class Weight_Signal(BaseSignal):
    def __init__(self,symbol, data, refdates):
        super().__init__(symbol, data, refdates)
        self.signal_name1 = 'Liquidity_Ratio'
        self.signal_name2 = 'MarketCap'
    
    def calculate_signal(self, refdate):
        lr = self.data['liquidity_ratio'].loc[refdate]
        mc = self.data['marketcap'].loc[refdate]
        return {self.signal_name1: lr, self.signal_name2 : mc}
    
