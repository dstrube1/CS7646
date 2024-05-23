"""
3 indicators:
1- Simple Moving Average => price / SMA
2- Bollinger Bands => Bollinger Bands Percentage TODO
3- Exponential Moving Average
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
    return 'dstrube3'


def bollinger_bands_percentage(df, days=10, plot=False, stocks=None):
    # Takes a dataframe and returns a dataframe with Bollinger Bands values for the given window

    # df = df / df.iloc[0, :]
    std_df = df.rolling(days).std()
    sma_df = df.rolling(days).mean()
    upper_band = sma_df + 2 * std_df
    lower_band = sma_df - 2 * std_df

    percentage = (df - lower_band) / (upper_band - lower_band)
    # if plot:
    #     bbp_df = pd.concat([lower_band, upper_band], axis=1)
    #     bbp_df.columns = ['Lower band', 'Upper band']
    #     bbp_df = pd.concat([df, bbp_df], axis=1)
    #     plt.figure('bb')
    #     ax = bb_df.plot(title="Bollinger Bands Percentage of {} (Window={} days)".format(stocks[0], days), fontsize=12)
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Bollinger Band Percentage')
    #     plt.grid(visible=True)
    #     plt.savefig('images/bollinger_bands_percentage.png')
    #     plt.clf()

    return percentage


def ema(df, days=10):
    # Takes a dataframe and returns a dataframe with Exponential Moving Average values for the given window

    # df = df / df.iloc[0, :]
    ema_df = df * 2

    for i in range(df.shape[0] - days + 1):
        temp_ema = df.iloc[i]
        for j in range(days):
            temp_ema = df.iloc[i + j] * (2.0 / (j + 2)) + temp_ema * (1 - 2.0 / (j + 2))
        ema_df.iloc[i + days - 1] = temp_ema
    ema_df.iloc[0: (days - 1)] = np.nan

    return ema_df


# def ppi(df, day1=12, day2=26, plot=False, stocks=None):
#     # Takes a dataframe and returns a dataframe with Percentage Price Indicator values for the given window
#
#     df = df / df.iloc[0, :]
#     ppi_df = ((ema(df, day1) - ema(df, day2)) / ema(df, day2)) * 100
#     ppi_df[0:day2] = np.nan
#     # if plot:
#     #     plt.figure('ppi')
#     #     ax = ppi_df.plot(title="Percentage Price Indicator of {} ({} days/{} days)".format(stocks[0], day1, day2),
#     #                      fontsize=12)
#     #     ax.set_xlabel('Date')
#     #     ax.set_ylabel('Percentage Price Indicator')
#     #     plt.grid(visible=True)
#     #     plt.savefig('images/ppi.png')
#     #     plt.clf()
#     return ppi_df


def price_sma_ratio(df, days=10, plot=False, combine=False, stocks=None):
    sma_df = sma(df, days, plot=False, combine=False, stocks=None)
    ratio = df / sma_df
    return ratio


def sma(df, days=10, plot=False, combine=False, stocks=None):
    # Takes a dataframe and returns a dataframe with Simple Moving Average values for the given window

    # df = df / df.iloc[0, :]
    sma_df = df.rolling(window=days, min_periods=days).mean()
    # df = pd.concat([df, sma_df], axis=1)
    # if combine and stocks is not None:
    #     # Normalize dataframe:
    #     df_columns = stocks
    #     df_columns.append('SMA')
    #     df.columns = df_columns

    # if plot:
    #     plt.figure('sma')
    #     ax = df.plot(title="Simple Moving Average of {} (Window={} days)".format(stocks[0], days), fontsize=12)
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Average Prices')
    #     plt.grid(visible=True)
    #     plt.savefig('images/sma.png')
    #     plt.clf()

    return sma_df


def standardize(df):
    return (df - np.mean(df)) / np.std(df)


def main(df=None, stocks=None):
    if df is None or stocks is None:
        return

    sma(df, days=10, plot=True, combine=True, stocks=stocks)
    bollinger_bands_percentage(df, days=30, plot=True, stocks=stocks)
    # ppi(df, day1=12, day2=26, plot=True, stocks=stocks)

    ema_9 = ema(df, days=9)
    ema_99 = ema(df, days=99)
    ema_df = pd.concat([ema_9, ema_99], axis=1)
    ema_df.columns = ['EMA9', 'EMA99']

    # plt.figure('ema')
    # ax = ema_df.plot(title='EMA 9 vs EMA 99', fontsize=12)
    # ax.set_xlabel('Date')
    # ax.set_ylabel('EMA Prices')
    # plt.grid(visible=True)
    # plt.savefig('images/ema.png')
    # plt.clf()


if __name__ == '__main__':
    # Without data, this will just return immediately
    main(df=None, stocks=None)
