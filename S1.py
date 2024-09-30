import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nsetools import Nse
from nsepy import get_history
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to fetch Nifty 50 data
def fetch_nifty_data(start_date='2023-01-01', end_date='2023-12-31'):
    nse = Nse()
    stock_data = get_history(symbol='NIFTY', start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))
    stock_data = pd.DataFrame(stock_data)
    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_data.dropna(inplace=True)
    return stock_data

# Function to prepare data for LSTM model
def prepare_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to build and train the LSTM model
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

# Function to plot predictions
def plot_predictions(data, predictions, scaler):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Close'], label='Actual Prices')
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
    plt.plot(future_dates, scaler.inverse_transform(predictions), label='Predicted Prices', color='red')
    plt.title('Nifty 50 Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main function
def main():
    # Fetch and preprocess data
    data = fetch_nifty_data()
    X, y, scaler = prepare_data(data)
    
    # Train LSTM model
    model = train_lstm_model(X, y)
    
    # Make future predictions
    last_data = np.array(data[['Close']].tail(60))
    last_data_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(last_data)
    X_future = np.array([last_data_scaled])
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
    future_predictions_scaled = model.predict(X_future)
    future_predictions = scaler.inverse_transform(future_predictions_scaled)
    
    # Plot predictions
    plot_predictions(data, future_predictions, scaler)

if __name__ == "__main__":
    main()
