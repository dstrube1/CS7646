"""
CS 7646: Machine Learning for Trading
PROJECT 6: INDICATOR EVALUATION
David Strube - dstrube3

Code implementing indicators as functions that operate on DataFrames. There is no defined API, but when it runs, the
main method should generate the charts that will illustrate indicators in the report.

"Develop and describe 5 technical indicators"
1- Simple Moving Average
2- Bollinger Bands
3- Momentum
4- Percentage Price Indicator
5- Exponential Moving Average
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
    return 'dstrube3'


def bollinger_bands(df, days, plot=False, stocks=None):
    # Takes a dataframe and returns a dataframe with Bollinger Bands values for the given window

    std_df = df.rolling(days).std()
    sma_df = df.rolling(days).mean()
    upper_band = sma_df + 2 * std_df
    lower_band = sma_df - 2 * std_df
    bb_df = pd.concat([lower_band, upper_band], axis=1)
    bb_df.columns = ['Lower band', 'Upper band']
    bb_df = pd.concat([df, bb_df], axis=1)

    if plot:
        plt.figure('bb')
        ax = bb_df.plot(title="Bollinger Bands of {} (Window={} days)".format(stocks[0], days), fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Bollinger Band prices')
        plt.grid(visible=True)
        plt.savefig('images/bollinger_band.png')
        plt.clf()

    return bb_df


def ema(df, days):
    # Takes a dataframe and returns a dataframe with Exponential Moving Average values for the given window

    ema_df = df * 2

    # TODO optimize this
    for i in range(df.shape[0] - days + 1):
        temp_ema = df.iloc[i]
        for j in range(days):
            temp_ema = df.iloc[i + j] * (2.0 / (j + 2)) + temp_ema * (1 - 2.0 / (j + 2))
        ema_df.iloc[i + days - 1] = temp_ema
    ema_df.iloc[0: (days - 1)] = np.nan

    return ema_df


def golden_cross(df, day1, day2, plot=False, stocks=None):
    # Takes a dataframe and returns a dataframe with Golden Cross values for the given window

    sma_day1 = sma(df, day1)
    sma_day2 = sma(df, day2)
    gc_df = pd.concat([sma_day1, sma_day2], axis=1)

    gc_df.columns = ["{}-day SMA".format(day1), "{}-day SMA".format(day2)]

    if plot:
        plt.figure('gc')
        ax = gc_df.plot(title="Golden Cross of {} ({} days/{} days)".format(stocks[0], day1, day2), fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('SMA difference between two windows')
        plt.grid(visible=True)
        plt.savefig('images/golden_cross.png')
        plt.clf()
    return gc_df


def momentum(df, days, plot=False, stocks=None):
    # Takes a dataframe and returns a dataframe with momentum values for the given window

    momentum_df = df / df.shift(days) - 1

    if plot:
        plt.figure('momentum')
        ax = momentum_df.plot(title="Momentum of {} (Window={} days)".format(stocks[0], days), fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Momentum ratio')
        plt.grid(visible=True)
        plt.savefig('images/momentum.png')
        plt.clf()
    return momentum_df


def ppi(df, day1=12, day2=26, plot=False, stocks=None):
    # Takes a dataframe and returns a dataframe with Percentage Price Indicator values for the given window

    ppi_df = ((ema(df, day1) - ema(df, day2)) / ema(df, day2)) * 100
    ppi_df[0:day2] = np.nan
    if plot:
        plt.figure('ppi')
        ax = ppi_df.plot(title="Percentage Price Indicator of {} ({} days/{} days)".format(stocks[0], day1, day2),
                         fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage Price Indicator')
        plt.grid(visible=True)
        plt.savefig('images/ppi.png')
        plt.clf()
    return ppi_df


def sma(df, days, plot=False, combine=False, stocks=None):
    # Takes a dataframe and returns a dataframe with Simple Moving Average values for the given window

    sma_df = df.rolling(days).mean()
    if combine and stocks is not None:
        # df = df / df.iloc[0, :]
        # Normalize something else:
        # sma_df = sma_df / sma_df.iloc[0, :]
        df = pd.concat([df, sma_df], axis=1)
        df_columns = stocks
        df_columns.append('SMA')
        df.columns = df_columns

    if plot:
        plt.figure('sma')
        ax = df.plot(title="Simple Moving Average of {} (Window={} days)".format(stocks[0], days), fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Prices')
        plt.grid(visible=True)
        plt.savefig('images/sma.png')
        plt.clf()

        # price_over_sma(df, sma_df, days, stocks)

    return sma_df


def price_over_sma(df, sma_df, days, stocks=None):
    quotient = df.copy()
    quotient /= sma_df
    df /= df.iloc[0, :]
    df = pd.concat([df, quotient], axis=1)
    df_columns = stocks
    df_columns.append('SMA')
    df_columns.append('Price / SMA')
    df.columns = df_columns

    plt.figure('price_over_sma')
    ax = df.plot(title="Price / SMA of {} (Window={} days)".format(stocks[0], days), fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Prices / SMA')
    plt.grid(visible=True)
    plt.savefig('images/price_over_sma.png')
    plt.clf()


def main(df=None, stocks=None):
    if df is None or stocks is None:
        return

    # TODO: Add None checks to all functions that are accessible outside this file (i.e., all functions here)

    sma(df, days=10, plot=True, combine=True, stocks=stocks)
    bollinger_bands(df, days=30, plot=True, stocks=stocks)
    momentum(df, days=10, plot=True, stocks=stocks)
    ppi(df, day1=12, day2=26, plot=True, stocks=stocks)
    golden_cross(df, day1=5, day2=20, plot=True, stocks=stocks)

    ema_12 = ema(df, days=12)
    ema_26 = ema(df, days=26)
    ema_df = pd.concat([ema_12, ema_26], axis=1)
    ema_df.columns = ['EMA12', 'EMA26']

    plt.figure('ema')
    ax = ema_df.plot(title='EMA 12 vs EMA 26', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('EMA Prices')
    plt.grid(visible=True)
    plt.savefig('images/ema.png')
    plt.clf()


if __name__ == '__main__':
    # Without data, this will just return immediately
    main(df=None, stocks=None)
