import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fetch_stock_data(ticker, period='1y'):
    stock_data = yf.download(ticker, period=period)
    return stock_data

def preprocess_data(stock_data):
    stock_data = stock_data.dropna()
    # Remove outliers
    stock_data = stock_data[(np.abs(stats.zscore(stock_data)) < 3).all(axis=1)]
    scaler = StandardScaler()
    stock_data[['Close', 'Volume']] = scaler.fit_transform(stock_data[['Close', 'Volume']])
    return stock_data
