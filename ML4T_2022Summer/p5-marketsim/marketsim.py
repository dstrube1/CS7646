""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""MC2-P1: Market simulator.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

Student Name: David strube
GT User ID: dstrube3
GT ID: 901081581
"""

import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data, get_orders_data_file
import scipy.optimize as spo


def get_orders_and_date_range(orders_file_name):
    orders = pd.read_csv(orders_file_name, index_col='Date', parse_dates=True, na_values=['nan'])

    # The start date and end date of the simulation are the first and last dates with orders in the orders_file.
    # (Note: The orders may not appear in sequential order in the file.)
    orders = orders.sort_index()

    date_range = pd.date_range(orders.index[0], orders.index[-1])

    return orders, date_range


def get_symbols_and_prices(orders, date_range):
    # From orders and date_range, get list of symbols and prices
    indexed_symbols = orders['Symbol']
    nondistinct_symbols = list(indexed_symbols)
    symbols = []
    for symbol in nondistinct_symbols:
        if symbol not in symbols:
            symbols.append(symbol)

    # Get prices, add column for cash, and fill in the gaps
    prices = get_data(symbols, date_range)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    return symbols, prices


def compute_portvals(
        orders_file='./orders/orders-02.csv',
        start_val=1000000.0,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
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
    # NOTE: orders_file may be a string, or it may be a file object.
    # Code should work correctly with either input.

    # Missing values
    if orders_file is None or start_val is None or commission is None or impact is None:
        return None
    # Invalid values?
    if commission < 0 or impact < 0 or commission > start_val:
        # Commission == 0? Sure, I can be doing this out of the goodness of my heart
        # Impact == 0? Sure, why not
        # Start value == 0? Sure, I can take some chances, bet a lot and risk bankruptcy - this is America
        # Negative commission, negative impact, or commission > start_val: does not compute
        # start_val < 0? Not sure
        return None
    if isinstance(orders_file, str):
        orders_file_name = orders_file
    else:
        # Assume if not string, then file
        # TODO: verify is file?
        orders_file_name = orders_file.name

    # Get orders and date range from orders_file_name
    # TODO: verify file_name is valid path?
    orders, date_range = get_orders_and_date_range(orders_file_name)

    # Get symbols and prices from orders and date_range
    # These prices won't get used until sub_test_code
    symbols, sub_test_code_prices = get_symbols_and_prices(orders, date_range)

    # Get prices
    prices = get_data(symbols, date_range, addSPY=False)
    prices.index.name = 'Date'
    prices = prices.dropna(axis=0, how='any')

    # Cash symbol
    cash = 'Cash'

    # Portfolio value symbol
    portval = 'portval'

    # Market Simulator lecture, 1:02:40; Happy birthday Mrs Balch!
    # https://edstem.org/us/courses/22242/lessons/33917/slides/196201
    # skip_date = dt.datetime(2011, 6, 15)
    # Son of a balch!, didn't see this until after the due date:
    # The “secret” regarding leverage and a “secret date” discussed in the YouTube lecture do not apply
    # and should be ignored.

    # Trades, holdings, and values
    thv = prices.copy()
    thv[:] = 0
    thv[cash] = start_val
    thv[portval] = start_val

    for i in range(orders.shape[0]):
        date = orders.index[i]
        # if date == skip_date:
        #    continue
        symbol = orders.iloc[i, 0]
        option = orders.iloc[i, 1]
        shares = orders.iloc[i, 2]
        price = prices.loc[date, symbol]

        if option == 'BUY':
            # Update cash
            thv.loc[date, cash] = thv.loc[date, cash] - shares * price * (impact + 1) - commission
            # Update shares
            thv.loc[date, symbol] = thv.loc[date, symbol] + shares
        else:  # If not buy, then sell
            # Update cash
            thv.loc[date, cash] = thv.loc[date, cash] + shares * price * (1 - impact) - commission
            # Update shares
            thv.loc[date, symbol] = thv.loc[date, symbol] - shares
        thv.loc[date:, cash] = thv.loc[date, cash]
        thv.loc[date:, symbol] = thv.loc[date, symbol]

    # At the end of trading, settle up holdings
    # Values = prices * holdings
    thv.loc[:, portval] = (thv.iloc[:, range(0, prices.shape[1])] * prices).sum(axis=1) + thv.loc[:, cash]

    portfolio_values = thv.loc[:, portval]
    return portfolio_values


def compute_daily_return(port_val):
    # From P2
    # Compute and return the daily return values, as shown in lesson 01-04, video 10 / 01-06, video 5
    """# Without Pandas:
    # Make a copy where we can save computed values
    #daily_returns = data_frame.copy()

    # To get the daily return of data for date at index t, divide the value at index t by the value at index t-1,
    # then subtract 1, starting with index 1
    #daily_returns[1:] = (data_frame[1:] / data_frame[:-1].values) - 1
    #daily_returns[1:] = (data_frame[1:] / data_frame[:-1]) - 1
    # Use .values here to access the underlying numpy array. When given two dataframes, pandas will try to match each
    # row based on index when performing elementwise arithmetic operations
    #"""

    # With pandas:
    daily_returns = (port_val / port_val.shift(1)) - 1

    # Set the values for row 0 to all zeroes (with or without pandas. Pandas leave the 0th row full of NaNs.)
    # daily_returns.iloc[0, :] = 0
    # Or better yet, just exclude the 0th row
    daily_returns = daily_returns[1:]

    return daily_returns


def compute_daily_portfolio_values(prices, allocations):
    # From P2
    # Assume prices have already been normalized
    prices = prices * allocations
    daily_portfolio_values = prices.sum(axis=1)
    return daily_portfolio_values


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


def compute_sharpe(daily_returns):
    # From P2
    # Sharpe ratio, as described in lesson 01-07, videos 6, 7, & 8
    # Named after William Sharpe: risk-adjusted return
    # Rp: portfolio return
    # Rf: Risk-free rate of return
    # sp: Standard deviation of portfolio return
    # One possible Sharpe ratio: (Rp - Rf) / sp
    # As proposed by William Sharpe: E[Rp - Rf] / std[Rp - Rf] (where E = expected value of return on the portfolio)
    # ex ante - forward-looking; we want to be back looking:
    # mean(daily_returns - daily_risk_free) / std(daily_returns - daily_risk_free)
    # = mean(daily_returns - daily_risk_free) / std(daily_returns),
    # because (standard deviation of range of values minus a constant) = standard deviation of range of values
    # daily_risk_free: LIBOR (London Interbank Offer Rate), 3moT-bill (3 month treasury bill interest rate), or 0%, or:
    # (1.1^(1/252)) - 1
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    # Not annualized:
    # sharpe_ratio = (daily_returns.mean() / daily_returns.std())
    return sharpe_ratio


def sub_test_code(orders_file_name):
    orders_file = open(orders_file_name)

    # Start value
    sv = 1000000

    # Process orders
    # Tested with both string and file for orders_file
    portvals = compute_portvals(orders_file=orders_file, start_val=sv)  # , impact=0)

    orders_file.close()

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        print('Warning: code did not return a DataFrame')
        return

    # Get portfolio stats
    orders, date_range = get_orders_and_date_range(orders_file_name)
    start_date = date_range[0]
    end_date = date_range[-1]

    daily_returns = compute_daily_return(portvals)
    symbols, prices = get_symbols_and_prices(orders, date_range)
    allocations = find_allocations(prices)
    daily_portfolio_values = compute_daily_portfolio_values(prices, allocations)
    cumulative_return = (daily_portfolio_values[-1] / daily_portfolio_values[0]) - 1
    average_daily_return = daily_returns.mean()
    standard_deviation_daily_return = daily_returns.std()
    sharpe_ratio = compute_sharpe(daily_returns)

    # TODO calculate these with the data I have
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Compare portfolio against $SPX
    # """
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cumulative_return}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {standard_deviation_daily_return}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {average_daily_return}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    # """
    print("Final Portfolio Value for", orders_file_name, ": $", end='')
    print("{:0,.2f}".format(float(portvals[-1])))


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during automatically grading his function will not be called.
    # Define input parameters
    dir_path = './orders/'
    dir_list = os.listdir(dir_path)
    dir_list.sort()
    for in_file in dir_list:
        orders_file_name = dir_path + in_file
        sub_test_code(orders_file_name)
    # sub_test_code("./orders/orders-11.csv")


def author():
    return 'dstrube3'


if __name__ == "__main__":
    test_code()
