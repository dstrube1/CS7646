"""
CS 7646: Machine Learning for Trading
PROJECT 6: INDICATOR EVALUATION
David Strube - dstrube3

Code implementing a TheoreticallyOptimalStrategy. It should implement testPolicy(), which returns a trades data frame

Theoretically Optimal Strategy only:

Allowable positions are 1,000 shares long, 1,000 shares short, 0 shares. (You may trade up to 2,000 shares at a time as
long as your positions are 1,000 shares long or 1,000 shares short.)

Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1,000 shares of JPM, and holding
that position.

Transaction costs for TheoreticallyOptimalStrategy:
Commission: $0.00
Impact: 0.00.
"""

import pandas as pd
import datetime as dt
import numpy as np
from util import get_data


def author():
    return 'dstrube3'


def testPolicy(symbol, sd, ed):
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates, addSPY=True, colname='Adj Close')
    prices = prices_all[symbol]

    prices2 = prices * 2

    for i in range(prices.shape[0] - 1):
        prices2.iloc[i+1] = prices.iloc[i+1] - prices.iloc[i]
    prices2[0] = np.nan

    action = np.sign(prices2.shift(-1)) * 1000
    trades = action * 2
    for i in range(action.shape[0] - 1):
        trades.iloc[i+1] = action.iloc[i+1] - action.iloc[i]
    trades[0] = np.nan
    trades.iloc[0] = action[0]
    trades.iloc[-1] = 0
    trades.columns = 'Shares'

    df_trades = pd.DataFrame(data=trades.values, index=trades.index, columns=['Shares'])

    return df_trades


if __name__ == '__main__':
    orders = testPolicy('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))
