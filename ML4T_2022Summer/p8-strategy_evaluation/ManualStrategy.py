import datetime as dt
import indicators as ind  # price_sma_ratio, bollinger_bands_percentage, ema
import marketsimcode as msc
from util import get_data
import pandas as pd


class ManualStrategy(object):
    """
    """
    # constructor
    def __init__(self, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose

    def author(self):
        return 'dstrube3'

    def testPolicy(
            self,
            symbol='JPM',
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100_000):
        if self.verbose:
            print(f"testPolicy: symbol: {symbol}, sd: {sd}, ed: {ed}, sv: {sv}")
        prices = get_data(symbol, pd.date_range(sd, ed))
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=False)
        price_sma_ratio = ind.price_sma_ratio(prices[symbol])
        bollinger_bands_percentage = ind.bollinger_bands_percentage(prices[symbol])
        ema = ind.ema(prices[symbol])
        dates = []
        orders = []
        shares = []
        position = 0
        for i in range(prices.shape[0]-1):
            direction = 0
            if price_sma_ratio[i] < 0.9 and bollinger_bands_percentage[i] < 0.2:
                direction = 1
            elif bollinger_bands_percentage[i] > 0.8 and ema[i] < 0:
                direction = -1
            # Always add a date
            dates.append(prices.index[i])
            if direction == 0:
                orders.append('HOLD')
                shares.append(0)
                # Leave position alone
            if direction == 1:
                if position == 0:
                    orders.append('BUY')
                    shares.append(1000)
                    position = 1000
                elif position == -1000:
                    orders.append('BUY')
                    shares.append(2000)
                    position = 1000
                elif position == 1000:
                    orders.append('HOLD')
                    shares.append(0)
                elif self.verbose:
                    print(f"WTH? Unexpected position value: {position}")
            if direction == -1:
                if position == 0:
                    orders.append('SELL')
                    shares.append(1000)
                    position = -1000
                elif position == 1000:
                    orders.append('SELL')
                    shares.append(2000)
                    position = -1000
                elif position == -1000:
                    orders.append('HOLD')
                    shares.append(0)
                    position = -1000
                elif self.verbose:
                    print(f"WTH? Unexpected position value: {position}")

        trades = pd.DataFrame(index=dates, columns=['Symbol', 'Order', 'Shares'])
        trades['Symbol'] = symbol
        trades['Order'] = orders
        trades['Shares'] = shares
        return trades

    def benchmark(
            self,
            symbol='JPM',
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100_000,
            commission=9.95,
            impact=0.005):
        prices = get_data(symbol, pd.date_range(sd, ed))
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=False)
        trades = pd.DataFrame(index=[prices.index[0], prices.index[-1]], columns=['Symbol', 'Order', 'Shares'])
        bench_port_val = msc.compute_portvals(trades, start_val=sv, commission=commission, impact=impact)
        return trades


def manual_strategy():
    symbol = 'JPM',
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100_000
    manualStrategy = ManualStrategy()
    df_trades = manualStrategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)


if __name__ == "__main__":
    manual_strategy()
