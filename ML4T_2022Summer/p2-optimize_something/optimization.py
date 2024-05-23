""""""
"""MC1-P2: Optimize a portfolio.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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

Student Name: David Strube
GT User ID: dstrube3
GT ID: 901081581
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo


def compute_daily_portfolio_values(prices, allocations):
    # Assume prices have already been normalized
    prices = prices * allocations
    daily_portfolio_values = prices.sum(axis=1)
    return daily_portfolio_values


def function_to_minimize(prices, initial_guess):
    port_val = compute_daily_portfolio_values(prices, initial_guess)
    daily_returns = compute_daily_return(port_val)
    sharpe_ratio = compute_sharpe(daily_returns)
    return -sharpe_ratio


def find_allocations(prices):
    num_stocks = prices.shape[1]
    initial_guess = np.array(num_stocks * [(1 / num_stocks)])
    result = spo.minimize(function_to_minimize, initial_guess, args=prices, method='SLSQP',
                          bounds=[(0, 1)] * num_stocks,
                          constraints=({'type': 'eq', 'fun': lambda initial_guess: 1 - np.sum(initial_guess)}),
                          options={'disp': True})
    allocations = result.x
    return allocations


def compute_daily_return(port_val):
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


def compute_sharpe(daily_returns):
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


def compute_stats(prices, allocations):
    # Portfolio statistics, as described in lesson 01-07, videos 2 & 3
    daily_returns = compute_daily_return(prices)

    daily_portfolio_values = compute_daily_portfolio_values(prices, allocations)

    cumulative_return = (daily_portfolio_values[-1] / daily_portfolio_values[0]) - 1

    average_daily_return = daily_returns.mean()

    standard_deviation_daily_return = daily_returns.std()

    # port_val = compute_daily_portfolio_values(prices, allocations)
    sharpe_ratio = compute_sharpe(daily_returns)

    return cumulative_return, average_daily_return, standard_deviation_daily_return, sharpe_ratio


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
    gen_plot=False,
):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range (start date & end date)
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    # Fill in the gaps, forward fill first
    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method="bfill", inplace=True)

    # Normalize
    prices_all = prices_all / prices_all.iloc[0, :]

    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY / S&P 500, for comparison later

    # find the allocations for the optimal portfolio
    allocs = find_allocations(prices)

    cr, adr, sddr, sr = compute_stats(prices, allocs)

    # Get daily portfolio value
    port_val = compute_daily_portfolio_values(prices, allocs)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # fig, ax = plt.subplots()

        df_temp = pd.concat([port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1)
        df_temp.plot()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title("Daily Portfolio Value and SPY")
        # plt.show()
        # plt.show break plt.savefig
        plt.savefig('images/Figure1-P6.png')

        """# create your base plot (for your report) using ax.plot()
        #   don't forget to save it using plt.savefig() first before adding the watermark to the existing axis

        # set the values of WATERMARK_TEXT and WATERMARK_FONT_SIZE (and file_name, of course) beforehand
        ax.text( # reference: https://matplotlib.org/3.5.0/gallery/text_labels_and_annotations/watermark_text.html
            0.5,
            0.5,
            'dstrube3@gatech.edu',
            transform=ax.transAxes,
            fontsize=20,
            color='gray',
            alpha=0.5,
            ha='center',
            va='center',
            rotation='30'
        )

        plt.savefig("images/Figure1_watermarked.png", bbox_inches="tight")
        # only share the "*_watermarked.png" images!"""

        plt.clf()

    return allocs, cr, adr, sddr, sr


def test_code():
    """
    This function WILL NOT be called by the auto grader.
    """

    start_date = dt.datetime(2008, 1, 1)  # dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 12, 31)  # dt.datetime(2009, 6, 1)
    symbols = ['JPM']  # ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
