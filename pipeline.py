# -*- coding: utf-8 -*-

"""
Feed data into return and signal
"""

import os
import pandas as pd
#from pandas.tseries.offsets import BDay, DateOffset
from signals_derivative import EWMA_Signal
from signals_derivative import Quantile_Signal
from signals_derivative import RSI_Signal, TurnoverRatio_Signal, Industry_Signal, HVIV_Signal, HVtrend_Signal,IVtrend_Signal, Weight_Signal
from shortreturn import shortVolPnL

class Pipeline:
    
    T = 0.5  # Assuming a constant value of T

    @staticmethod
    def align_and_add_refdates(df, refdates):
        # Reindex the DataFrame to include the nearest valid dates for refdates
        df_reindexed = df.reindex(refdates, method='nearest')
        
        # Combine the original DataFrame with the reindexed DataFrame
        df_combined = pd.concat([df, df_reindexed])
        
        # Drop duplicate rows based on the index
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        # Sort the DataFrame by index (dates)
        df_combined = df_combined.sort_index()
        
        return df_combined    
    
    @staticmethod
    def generate_y(refdates, ivs, ives, rvs, realivs):
        returns_df = pd.DataFrame(index=refdates, columns=['return','pnl'])
        returns_comp = pd.DataFrame(index = refdates, columns = ['return', 'iv' ,'ive', 'rv', 'realiv'])
        for refdate in refdates:
            iv = ivs[refdate]
            ive = ives[refdate]
            rv = rvs[refdate]
            realiv = realivs[refdate]
            
            pnlo = shortVolPnL(Pipeline.T, iv, ive, rv, realiv)
            returns_df.loc[refdate, 'return'] = pnlo.LT_Return()
            returns_df.loc[refdate, 'pnl'] = pnlo.LT_PnL()
            returns_comp.loc[refdate, 'iv'] = iv
            returns_comp.loc[refdate, 'ive'] = ive
            returns_comp.loc[refdate, 'rv'] = rv
            returns_comp.loc[refdate, 'realiv'] = realiv
        return returns_df, returns_comp
        
    @staticmethod
    def process_ticker_data(ticker_file, refdatesall):
        df = pd.read_csv(ticker_file, index_col='PriceDate', parse_dates=['PriceDate'])
        ticker = df['symbol'].values[0]
        refdates = refdatesall[ticker]
        refdates_X = refdates[:-1]
        refdate_test = [refdates[-1]]
        df =Pipeline.align_and_add_refdates(df, refdates_X)
        signals = []
        params = (ticker, df, refdates_X)
        #Add your signals here
        signal_generators = [
            EWMA_Signal(*params),
            Quantile_Signal(*params),
            RSI_Signal(*params),
            TurnoverRatio_Signal(*params),
            Industry_Signal(*params),
            HVIV_Signal(*params),
            HVtrend_Signal(*params),
            IVtrend_Signal(*params),
            Weight_Signal(*params)
            
        ]
        
        for signal_gen in signal_generators:
            signal_df = signal_gen.get_signals()
            if signal_df.empty:
                continue
            signals.append(signal_df)
        
        if not signals:
            return pd.DataFrame(), pd.DataFrame()
        
        df_signals = pd.concat(signals, axis = 1).loc[:, ~pd.concat(signals, axis=1).columns.duplicated()]
        ivs = df.loc[refdates_X]['impvol180']
        ives = df.loc[refdates_X]['IVE']
        rvs = df.loc[refdates_X]['Forward_RV']
        realivs = df.loc[refdates_X]['impvol180theta']
        ###
        nan_refdates = realivs[realivs.isna()].index
        if not nan_refdates.empty:
            print(f"{ticker} : Refdates with NaN values in 'impvol180theta':")
            print(nan_refdates)
        
        
        y, y_comp = Pipeline.generate_y(refdates_X, ivs, ives, rvs, realivs)
        
        y_comp.to_csv(os.path.join(r'C:\GSaTasks\Others\VolStrategy1\return_output', f"{ticker}_components.csv"))
        df_train = df_signals.join(y)
        
        #test signal generation
        signals_test = []
        params_test = (ticker, df, refdate_test)
        signal_generators_test = [
            EWMA_Signal(*params_test),
            Quantile_Signal(*params_test),
            RSI_Signal(*params_test),
            TurnoverRatio_Signal(*params_test),
            Industry_Signal(*params_test),
            HVIV_Signal(*params_test),
            HVtrend_Signal(*params_test),
            IVtrend_Signal(*params_test),
            Weight_Signal(*params_test)
        ]
        
        for signal_gen in signal_generators_test:
            signal_df_test = signal_gen.get_signals()
            signals_test.append(signal_df_test)
        
        df_test= pd.concat(signals_test, axis=1).loc[:, ~pd.concat(signals_test, axis=1).columns.duplicated()]
        #for test df, I wanna only join the common rows
        
        
        return df_train, df_test
      
    @staticmethod
    def run(data_folder, refdatesall):
        db_train, db_test = [], []
        for ticker in refdatesall:
            ticker_path = os.path.join(data_folder, ticker + '.csv')
            df_train_ticker, df_test_ticker = Pipeline.process_ticker_data(ticker_path, refdatesall)
            if df_train_ticker.empty and df_test_ticker.empty:
                continue
                
            db_train.append(df_train_ticker)
            db_test.append(df_test_ticker)
        if not db_train:
             raise ValueError("No data was processed. The db_train list is empty.")
        db_train = pd.concat(db_train)
        db_test = pd.concat(db_test)
        
        return db_train, db_test

