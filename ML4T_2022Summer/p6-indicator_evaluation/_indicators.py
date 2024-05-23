"""
CS 7646: Machine Learning for Trading
PROJECT 6: INDICATOR EVALUATION
David Strube - dstrube3

Code implementing indicators as functions that operate on DataFrames. There is no defined API, but when it runs, the
main method should generate the charts that will illustrate indicators in the report.
"""

import pandas as pd


def author():
    return 'dstrube3'


def sma(df, window=5, bollinger=False, threshold=2):
    # Takes a dataframe and returns a dataframe with simple moving average values for the given window
    # Optionally adds Bollinger Bands as a tuple (mean, upper-band, lower-band)

    mean = df.rolling(window, center=False).mean()

    if bollinger:
        std = pd.rolling_std(df, window)
        upper = mean + threshold * std
        upper.rename(columns={mean.columns[0]: 'upper_band'}, inplace=True)

        lower = mean - threshold * std
        lower.rename(columns={mean.columns[0]: 'lower_band'}, inplace=True)

        mean.rename(columns={mean.columns[0]: 'mean'}, inplace=True)
        return mean, upper, lower

    mean.rename(columns={mean.columns[0]: 'mean'}, inplace=True)
    return mean


def ema(df, days=12):
    # Takes a dataframe and number of days and returns an exponential moving average
    ema = pd.ewma(df.copy(), com=((days - 1) / 2))
    ema.rename(columns={ema.columns[0]: 'exp_ma'}, inplace=True)
    return ema


def macd(df, ema1_days=12, ema2_days=26, macd_signal_days=9):
    # Accepts a dataframe, returns a dataframe with moving average convergence/divergence - MACD_signal,
    # where MACD is (ema1-ema2) and MACD_signal is the EMA of MACD
    ema1 = ema(df, ema1_days)
    ema2 = ema(df, ema2_days)

    macd = ema1 - ema2
    macd_signal = ema(macd, macd_signal_days)

    macd_signal.rename(columns={macd.columns[0]: 'macd_signal'}, inplace=True)
    macd.rename(columns={macd.columns[0]: 'macd'}, inplace=True)
    return macd, macd_signal


def stoc_osc(df, k_window=14, d_window=3):
    # Returns the current market rate of the main and average lines respectively
    high = pd.rolling_max(df, window=k_window)
    low = pd.rolling_min(df, window=k_window)
    K = 100 * (df - low) / (high - low)
    D = pd.rolling_mean(K, window=d_window)

    K.rename(columns={K.columns[0]: 'K'}, inplace=True)
    D.rename(columns={D.columns[0]: 'D'}, inplace=True)
    return K, D


