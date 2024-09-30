import argparse
import pandas as pd
import numpy as np
from nsetools import Nse
from nsepy import get_history # Import get_history for fetching data
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, LSTM  # type: ignore
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Function to fetch stock data from NSE
def fetch_stock_data(ticker, period='1y', data_type='stock'):
    nse = Nse()
    # Use 'NIFTY 50' as the symbol instead of ticker for fetching data
    stock_data = get_history(symbol='NIFTY 50', start=pd.to_datetime('2023-01-01'), end=pd.to_datetime('2023-12-31'))
    stock_data = pd.DataFrame(stock_data)
    # Check if 'Date' is already in the index
    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)  # Move date from index to column if necessary
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    return stock_data

# Function to preprocess data
def preprocess_data(data):
    # Check the shape of the data before and after dropna() to see if all rows are removed
    print("Shape before dropna:", data.shape)
    data = data.dropna()
    print("Shape after dropna:", data.shape)
    return data

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data = data.dropna()
    return data

# Function to prepare LSTM data
def prepare_lstm_data(data, window_size=20):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function for backtesting strategy
def backtest_strategy(data):
    # Implement backtesting logic
    print("Backtesting strategy...")

# Function to evaluate model
def evaluate_model(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))
    print(f"Mean Squared Error: {mse}")

# Function to plot predictions
def plot_predictions(data, predictions, scaler):
    plt.figure(figsize=(14,5))
    plt.plot(data['Close'].values, color='blue', label='Actual Stock Price')
    plt.plot(np.concatenate([np.zeros(len(data) - len(predictions)), predictions]), color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function to fetch news sentiment (placeholder)
def fetch_news_sentiment(ticker):
    # Implement sentiment analysis logic
    return np.random.rand() * 2 - 1  # Placeholder: random sentiment

# Function for portfolio optimization (placeholder)
def optimize_portfolio(predictions, data):
    expected_return = np.mean(predictions)
    risk = np.std(predictions)
    return expected_return, risk

# Function for risk management (placeholder)
def risk_management(predictions):
    VaR = np.percentile(predictions, 5)  # 5th percentile as a simple VaR calculation
    return {"Value at Risk": VaR}

# Function to create an interactive stock price chart
def plot_stock_chart(stock_data):
    fig = go.Figure()

    # Plotting the closing price
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))

    # Plotting technical indicators
    if 'MA20' in stock_data.columns:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA20'], mode='lines', name='MA20'))
    if 'MA50' in stock_data.columns:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='MA50'))

    fig.update_layout(title='Stock Price and Technical Indicators', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to create a candlestick chart
def plot_candlestick_chart(stock_data):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    )])

    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to create a sentiment analysis chart
def plot_sentiment(sentiments):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sentiments.index, sentiments.values, marker='o', linestyle='-', color='b')
    ax.set_title('Sentiment Analysis Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')

    # Convert matplotlib figure to PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return f"data:image/png;base64,{img_str}"

# Function to create a dashboard using Dash
def create_dashboard(stock_data, sentiments):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Stock Market Dashboard"),
        dcc.Graph(figure=plot_stock_chart(stock_data), id='stock-chart'),
        dcc.Graph(figure=plot_candlestick_chart(stock_data), id='candlestick-chart'),
        html.Img(src=plot_sentiment(sentiments), style={'width': '100%', 'height': 'auto'}),
    ])

    app.run_server(debug=True)

def main(ticker, period='1y', data_type='stock'): # type: ignore
    # Data Preparation
    print(f"Fetching data for {ticker} with period {period}...")
    stock_data = fetch_stock_data(ticker, period, data_type)
    stock_data = preprocess_data(stock_data)
    stock_data = calculate_technical_indicators(stock_data)
    
    # News Sentiment Analysis
    sentiment = fetch_news_sentiment(ticker)
    print(f"Average sentiment for {ticker}: {sentiment}")
    
    # Model Preparation and Training
    X, y, scaler = prepare_lstm_data(stock_data)
    model = build_lstm_model(X.shape)
    model.fit(X, y, epochs=50, batch_size=32)
    
    # Predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Evaluation
    evaluate_model(y, predictions)
    
    # Backtesting
    backtest_strategy(stock_data)
    
    # Visualization
    plot_predictions(stock_data, predictions, scaler)
    
    # Portfolio Management
    expected_return, risk = optimize_portfolio(predictions, stock_data)
    print(f"Expected Return: {expected_return}, Risk: {risk}")
    
    # Risk Management
    risk_metrics = risk_management(pd.Series(predictions))
    print(f"Risk Metrics: {risk_metrics}")
    
    # Enhanced Visualization and Dashboard
    create_dashboard(stock_data, pd.Series([sentiment] * len(stock_data), index=stock_data.index))

def main(ticker, period='1y', data_type='stock'):
    # Data Preparation
    print(f"Fetching data for {ticker} with period {period}...")
    stock_data = fetch_stock_data(ticker, period, data_type)
    stock_data = preprocess_data(stock_data)
    # Check if the DataFrame is empty after preprocessing
    if stock_data.empty:
        print("Error: DataFrame is empty after preprocessing.")
        return  # Exit the program if the DataFrame is empty

    stock_data = calculate_technical_indicators(stock_data)
