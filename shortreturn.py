# -*- coding: utf-8 -*-
"""
Assuming dividend payment is less than Time Value, so American option can be priced using BSM
fix vega is 10000
"""

import numpy as np
import scipy.stats as si


class shortVolPnL:
    def __init__(self, T, iv, ive, RV, RealIV, vega = 10000):
        """
        Initialize the ShortVolPnL class with given parameters.

        :param S: Spot price of the underlying asset
        :param K: Strike price of the option
        :param T: Time to expiration in years
        :param r: Risk-free interest rate
        :param iv: Implied volatility of the option when entrying the position
        :param ive: IV at the end of the trading window
        :vega: fix to short 10000 vega
        """
        self.T = T
        self.iv = iv
        self.ive = ive
        self.rv = RV
        self.vega = vega
        self.RealIV = RealIV
    def LT_Return(self, option_type = "Call"):
        """
        Calculate the PnL of a short volatility strategy.

        :param option_type: Type of the option ('call' or 'put')
        :return: 'Delta-Residualized' standardlized Return of the short volatility strategy
        """
        theta = - self.vega * self.RealIV/(2*self.T) #decay using the realIV(rate)
        fr_pnl = - ((self.RealIV**2 - self.rv**2) / self.RealIV**2) * theta
        pnl_vega = (self.iv - self.ive) * self.vega
        return (fr_pnl + pnl_vega)/ self.RealIV #risk averse
    
    def LT_PnL(self, option_type = "Call"):
        theta = - self.vega * self.RealIV/(2*self.T)
        fr_pnl = - ((self.RealIV**2 - self.rv**2) / self.RealIV**2) * theta
        pnl_vega = (self.iv - self.ive) * self.vega
        
        return (fr_pnl + pnl_vega)
        
    
    

    
    
    


