import ta

def calculate_technical_indicators(stock_data):
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()
    stock_data['MACD'] = ta.trend.MACD(stock_data['Close']).macd_diff()
    stock_data['Bollinger Bands'] = ta.volatility.BollingerBands(stock_data['Close']).bollinger_mavg()
    stock_data['ATR'] = ta.volatility.AverageTrueRange(stock_data['High'], stock_data['Low'], stock_data['Close']).average_true_range()
    return stock_data
