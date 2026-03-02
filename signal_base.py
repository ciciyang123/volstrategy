# -*- coding: utf-8 -*-

"""
Base class for signals
"""

import pandas as pd


# signal_base.py
class BaseSignal:
    def __init__(self, symbol, data, refdates):
        self.symbol = symbol
        self.data = data
        self.refdates = refdates

    def calculate_signal(self, refdate):
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_signals(self):
        signals = []
        for refdate in self.refdates:
            try:
                signal_value = self.calculate_signal(refdate)
                if signal_value is not None:
                    signal_data = {'Symbol': self.symbol, 'Date': refdate}
                    signal_data.update(signal_value)
                    signals.append(signal_data)
                else:
                    print(f"signal_value is {signal_value}")
            except Exception as e:
                print(f"Error calculating {self.symbol} signal for {refdate}: {e}")
        if not signals:
            return pd.DataFrame()
        return pd.DataFrame(signals).set_index('Date')
    
    
