"""
CS 7646: Machine Learning for Trading
PROJECT 6: INDICATOR EVALUATION
David Strube - dstrube3

An improved version of marketsim code accepts a “trades” DataFrame (instead of a file). It is OK not to submit this
file if I have subsumed its functionality into one of my other required code files. This file has a different name and
a slightly different setup than my previous project. However, that solution can be used with several edits for the new
requirements.
"""

import datetime as dt
import TheoreticallyOptimalStrategy as tos
import pandas as pd
from util import get_data
import numpy as np
import scipy.optimize as spo


def author():
    return 'dstrube3'


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

    daily_returns.iloc[0] = 0

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


def compute_portvals(orders, start_val=100_000.0, commission=0.0, impact=0.0):
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

    orders = orders.sort_index()
    start_date = orders.index.min()
    end_date = orders.index.max()
    orders.insert(0, 'Symbol', 'JPM')
    orders.insert(1, 'Order', 'SELL')
    for i in range(orders.shape[0]):
        if orders.iloc[i, 2] > 0:
            orders.iloc[i, 1] = 'BUY'
    orders['Shares'] = orders['Shares'].abs()
    orders = orders.loc[(orders.Shares != 0)]

    # Cash symbol
    cash = 'Cash'

    # Portfolio value symbol
    portval = 'portval'

    buying_table = get_data(orders['Symbol'].unique().tolist(), pd.date_range(start_date, end_date), addSPY=False)
    buying_table.index.name = 'Date'
    buying_table = buying_table.dropna(axis=0, how='any')
    holding_table = buying_table.copy()
    holding_table[:] = 0
    holding_table[cash] = start_val
    holding_table[portval] = start_val

    for i in range(orders.shape[0]):
        purchase_date = orders.index[i]
        symbol = orders.iloc[i, 0]
        option = orders.iloc[i, 1]
        shares = orders.iloc[i, 2]
        price = buying_table.loc[purchase_date, symbol]

        if option == 'BUY':
            holding_table.loc[purchase_date, cash] = holding_table.loc[purchase_date, cash] - shares * price * (
                        1 + impact) - commission
            holding_table.loc[purchase_date, symbol] = holding_table.loc[purchase_date, symbol] + shares
        else:
            holding_table.loc[purchase_date, cash] = holding_table.loc[purchase_date, cash] + shares * price * (
                        1 - impact) - commission
            holding_table.loc[purchase_date, symbol] = holding_table.loc[purchase_date, symbol] - shares
        holding_table.loc[purchase_date:, cash] = holding_table.loc[purchase_date, cash]
        holding_table.loc[purchase_date:, symbol] = holding_table.loc[purchase_date, symbol]
    holding_table.loc[:, portval] = (holding_table.iloc[:, range(0, buying_table.shape[1])] * buying_table).sum(
        axis=1) + holding_table.loc[:, cash]
    portvals = holding_table.loc[:, portval]

    return portvals


def test_code():
    start_value = 100_000
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    commission = 0.0
    impact = 0.0

    # Get theoretically optimal stuff
    orders = tos.testPolicy(symbol='JPM', sd=start_date, ed=end_date)

    # Process orders, see if it crashes
    portvals = compute_portvals(orders, start_val=start_value, commission=commission, impact=impact)

    if portvals is None:
        print('Problem with portvals')


if __name__ == '__main__':
    test_code()
