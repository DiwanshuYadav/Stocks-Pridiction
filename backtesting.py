import backtrader as bt

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(period=15)

    def next(self):
        if self.sma > self.data.close:
            self.sell()
        elif self.sma < self.data.close:
            self.buy()

def backtest_strategy(stock_data):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=stock_data)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.run()
    cerebro.plot()
