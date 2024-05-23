"""
CS 7646: Machine Learning for Trading
PROJECT 6: INDICATOR EVALUATION
David Strube - dstrube3

An improved version of marketsimcode accepts a “trades” DataFrame (instead of a file). It is OK not to submit this
file if I have subsumed its functionality into one of my other required code files. This file has a different name and
a slightly different setup than my previous project. However, that solution can be used with several edits for the new
requirements.
"""

import datetime as dt
import pandas as pd
from util import get_data
import scipy.optimize as spo
import numpy as np


def author():
    return 'dstrube3'


def get_symbols_and_date_range(trades):
    # From trades dataframe
    # get list of symbols and date_range
    indexed_symbols = trades['Symbol']
    nondistinct_symbols = list(indexed_symbols)
    symbols = []
    for symbol in nondistinct_symbols:
        if symbol not in symbols:
            symbols.append(symbol)

    # Assuming index column is Date, but not assuming it's sorted:
    trades = trades.sort_index()
    date_range = pd.date_range(trades.index[0], trades.index[-1])

    return symbols, date_range


def compute_daily_portfolio_values(prices, allocations):
    # From P2
    # Assume prices have already been normalized
    prices = prices * allocations
    daily_portfolio_values = prices.sum(axis=1)
    return daily_portfolio_values


def compute_sharpe(daily_returns):
    # From P2
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    return sharpe_ratio


def compute_daily_return(port_val):
    # From P2
    daily_returns = (port_val / port_val.shift(1)) - 1

    daily_returns = daily_returns[1:]

    return daily_returns


def function_to_minimize(prices, initial_guess):
    # From P2
    port_val = compute_daily_portfolio_values(prices, initial_guess)
    daily_returns = compute_daily_return(port_val)
    sharpe_ratio = compute_sharpe(daily_returns)
    return -sharpe_ratio


def find_allocations(prices):
    # From P2
    num_stocks = prices.shape[1]
    initial_guess = np.array(num_stocks * [(1 / num_stocks)])
    result = spo.minimize(function_to_minimize, initial_guess, args=prices, method='SLSQP',
                          bounds=[(0, 1)] * num_stocks,
                          constraints=({'type': 'eq', 'fun': lambda initial_guess: 1 - np.sum(initial_guess)}),
                          options={'disp': True})
    allocations = result.x
    return allocations


def compute_portvals(
        orders,
        start_val=1_000_000.0,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders: Trades dataframe
    :type orders: Dataframe
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading
    day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code

    # Missing values
    if orders is None or start_val is None or commission is None or impact is None:
        return None
    # Invalid values?
    if commission < 0 or impact < 0 or commission > start_val:
        # Commission == 0? Sure, I can be doing this out of the goodness of my heart
        # Impact == 0? Sure, why not
        # Start value == 0? Sure, I can take some chances, bet a lot and risk bankruptcy - this is America
        # Negative commission, negative impact, or commission > start_val: does not compute
        # start_val < 0? Not sure
        return None

    symbols, date_range = get_symbols_and_date_range(orders)

    # Cash symbol
    cash = 'Cash'

    # Prices data frame
    prices = get_data(symbols, date_range, addSPY=True, colname='Adj Close')
    prices[cash] = 1.0
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=False)

    # Trades data frame
    trades = prices.copy()
    trades[symbols] = 0
    trades[cash] = 0
    trades['SPY'] = 0

    for index, row in orders.iterrows():
        if index not in prices.index:
            continue
        date = index
        symbol = row['Symbol']
        order = row['Order']
        share = row['Shares']
        price = prices.ix[date][symbol]

        if order == 'SELL':
            trades.ix[date, cash] -= commission
            trades.ix[date, cash] += share * price * (1 - impact)
            trades.ix[date, symbol] -= share

        if order == 'BUY':
            trades.ix[date, cash] -= commission
            trades.ix[date, cash] -= share * price * (1 + impact)
            trades.ix[date, symbol] += share

    # Holdings data frame
    holdings = trades.copy()
    holdings.ix[0, cash] = holdings.ix[0, cash] + start_val
    for i in range(1, len(holdings)):
        holdings.ix[i] += holdings.ix[i - 1]

    # Values data frame
    portvals = pd.DataFrame(index=holdings.index)
    portvals['Total Value'] = 0
    for index, row in holdings.iterrows():
        for symbol in holdings.columns:
            portvals.ix[index, 'Total Value'] += row[symbol] * prices.ix[index, symbol]
    return portvals


def testCode():
    start_value = 100_000
    symbol = np.array(['JPM'])
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    date = pd.date_range(start_date, end_date)
    commission = 0.0
    impact = 0.0

    benchmark_values = get_data(symbol, date)
    benchmark_values = benchmark_values[symbol]

    benchmark_trades = np.zeros_like(benchmark_values)

    bm_portvals = compute_portvals(benchmark_trades, start_value, commission, impact)
    print("done")


if __name__ == '__main__':
    testCode()
