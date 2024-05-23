"""
CS 7646: Machine Learning for Trading
PROJECT 6: INDICATOR EVALUATION
David Strube - dstrube3

This file should be considered the entry point to the project. The if __name__ == '__main__': section of the code
will call the testPolicy function in TheoreticallyOptimalStrategy, as well as indicators and marketsimcode as
needed, to generate the plots and statistics for a report
"""

import pandas as pd
from util import get_data
import datetime as dt
import matplotlib.pyplot as plt
import indicators as ind
import TheoreticallyOptimalStrategy as tos
import marketsimcode as ms


def author():
    return 'dstrube3'


def main():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    stocks = ['JPM']

    df_trades = tos.testPolicy(symbol=stocks[0], sd=start_date, ed=end_date)

    # Theoretically Optimal Strategy
    tos_portvals = ms.compute_portvals(df_trades, start_val=100_000)

    dates = pd.date_range(start_date, end_date)
    benchmark = get_data(stocks, dates, addSPY=False, colname='Adj Close')
    bm_and_tos = benchmark.join(tos_portvals, how='outer')
    bm_and_tos.dropna(inplace=True)
    bm_and_tos_normalized = bm_and_tos / bm_and_tos.iloc[0, :]

    plt.figure('tos')
    ax = bm_and_tos_normalized.plot(title="Theoretically Optimal Strategy for {}".format(stocks[0]),
                                    fontsize=12, color=['purple', 'red'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    plt.savefig("images/tos.png")
    plt.clf()

    # Get stats
    bm_norm = bm_and_tos.iloc[:, 0]
    tos_norm = bm_and_tos.iloc[:, 1]
    bm_daily_returns = ms.compute_daily_return(bm_norm)
    tos_daily_returns = ms.compute_daily_return(tos_norm)
    bm_avg_daily_ret = bm_daily_returns.mean(axis=0)
    tos_avg_daily_ret = tos_daily_returns.mean(axis=0)
    bm_cum_ret = (bm_norm.iloc[-1] / bm_norm.iloc[0]) - 1
    tos_cum_ret = (tos_norm.iloc[-1] / tos_norm.iloc[0]) - 1
    bm_std_daily_ret = bm_norm.std(axis=0)
    tos_std_daily_ret = tos_norm.std(axis=0)

    # Print stats to file
    file = open('p6_results.txt', 'w')
    file.write("Benchmark Strategy Returns:\n" +
               "Benchmark Strategy Cumulative Return: {:0,.6f}".format(bm_cum_ret) + "\n" +
               "Benchmark Strategy Average Daily Return: {:0,.6f}".format(bm_avg_daily_ret) + "\n" +
               "Benchmark Strategy Standard Deviation Daily Return: {0:.6f}".format(bm_std_daily_ret) + "\n" +
               "Benchmark Strategy Normalized Ending Value: {}".format(bm_norm[-1]) + "\n" +
               "\n" +
               "Optimal Strategy Returns:\n" +
               "Optimal Strategy Cumulative Return: {0:.6f}".format(tos_cum_ret) + "\n" +
               "Optimal Strategy Average Daily Return: {0:.6f}".format(tos_avg_daily_ret) + "\n" +
               "Optimal Strategy Standard Deviation Daily Return: {0:.6f}".format(tos_std_daily_ret) + "\n" +
               "Optimal Strategy Normalized Ending Value: {}".format(tos_norm[-1]))
    file.close()

    # Get the dataframe to demonstrate the indicators
    df = get_data(stocks, pd.date_range(start_date, end_date), addSPY=False)
    df = df.dropna()
    # This runs the indicator functions and generates the charts from the functions' findings:
    ind.main(df=df, stocks=stocks)


if __name__ == '__main__':
    main()
